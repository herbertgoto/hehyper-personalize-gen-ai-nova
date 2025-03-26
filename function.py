import json
import boto3
import base64
import io
from io import BytesIO
from botocore.exceptions import ClientError
import uuid
# External dependencies
from PIL import Image
from urllib.parse import urlparse
import csv

boto3_bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')



def upload_pil_image_to_s3(image, bucket_name, object_key):
    # Create an S3 client
    s3_client = boto3.client('s3')

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    object_key += ".png"

    # Upload the image to S3
    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=img_byte_arr)
        print(f"Image uploaded successfully to {bucket_name}/{object_key}")
        return f"{object_key}"
    except Exception as e:
        print(f"Error uploading image to S3: {str(e)}")
        
def generate_uuid_filename():

    
    # Generate a UUID
    unique_id = uuid.uuid4()
    
    # Create the new filename
    new_filename = f"{unique_id}"
    
    return new_filename        

def get_image_froms3(event):
    s3_uri = event['s3_uri']
    

    # Parse the S3 URI to get the bucket name and object key
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        
        # Read the image data
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))
        return image
        
    except Exception as e:
        # If an error occurs, print the error message and return an error response
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error loading image: {str(e)}"
        }



def image_to_base64(img) -> str:
    """Converts a PIL Image or local image file path to a base64 string"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")


def call_image_gen(prompt, img1_b64, output_bucket):
    # Payload creation
    negative_prompts = 'poor quality, low resolution'
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText" : negative_prompts,
            "conditionImage": img1_b64,
            "controlMode": "CANNY_EDGE",# [OPTIONAL] CANNY_EDGE | SEGMENTATION. DEFAULT: CANNY_EDGE
            "controlStrength": 0.9
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    })


    # Model invocation
    response = boto3_bedrock.invoke_model(
        body = body, 
        modelId = "amazon.titan-image-generator-v2:0",
        accept = "application/json", 
        contentType = "application/json"
    )
    
    # Output processing
    response_body = json.loads(response.get("body").read())
    out_img_b64 = response_body["images"][0]
    img_data = base64.b64decode(out_img_b64)
    img_buffer = BytesIO(img_data)
    img_out = Image.open(img_buffer)
    
    print(out_img_b64)

    #return out_img_b64
    filename = generate_uuid_filename()
    return upload_pil_image_to_s3(img_out,output_bucket,filename)

def call_image_gen_amazon_nova(prompt, img1_b64, output_bucket):
    # Payload creation
    negative_prompts = 'poor quality, low resolution'
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText" : negative_prompts,
            "conditionImage": img1_b64,
            "controlMode": "CANNY_EDGE",# [OPTIONAL] CANNY_EDGE | SEGMENTATION. DEFAULT: CANNY_EDGE
            #"controlStrength": 0.9
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    })

    # Model invocation
    response = boto3_bedrock.invoke_model(
        body = body, 
        modelId = "amazon.nova-canvas-v1:0",
        accept = "application/json", 
        contentType = "application/json"
    )
    
    # Output processing
    response_body = json.loads(response.get("body").read())
    out_img_b64 = response_body["images"][0]
    img_data = base64.b64decode(out_img_b64)
    img_buffer = BytesIO(img_data)
    img_out = Image.open(img_buffer)
    
    print(out_img_b64)

    #return out_img_b64
    filename = 'nova_'+generate_uuid_filename()
    return upload_pil_image_to_s3(img_out,output_bucket,filename)
    
def get_completion(prompt, system_prompt=None):
    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    # Define the inference configuration
    inference_config = {
        "temperature": 0.0,  # Set the temperature for generating diverse responses
        "maxTokens": 4000,  # Set the maximum number of tokens to generate
        "topP": 1,  # Set the top_p value for nucleus sampling
    }
    # Create the converse method parameters
    converse_api_params = {
        "modelId": modelId,  # Specify the model ID to use
        "messages": [{"role": "user", "content": [{"text": prompt}]}],  # Provide the user's prompt
        "inferenceConfig": inference_config,  # Pass the inference configuration
    }
    # Check if system_text is provided
    if system_prompt:
        # If system_text is provided, add the system parameter to the converse_params dictionary
        converse_api_params["system"] = [{"text": system_prompt}]

    # Send a request to the Bedrock client to generate a response
    try:
        response = boto3_bedrock.converse(**converse_api_params)

        # Extract the generated text content from the response
        text_content = response['output']['message']['content'][0]['text']

        # Return the generated text content
        return text_content

    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")

def s3_json_uploader(bucket_name, jsonData):
    s3_client = boto3.client('s3')
    filename = generate_uuid_filename()
    # Upload the image to S3
    filename += ".json"
    try:
        s3_client.put_object(Bucket=bucket_name, Key=filename, Body=jsonData)
        print(f"Image uploaded successfully to {bucket_name}/{filename}.json")
        return f"{bucket_name}/{filename}"
    except Exception as e:
        print(f"Error uploading image to S3: {str(e)}")

def load_json_from_s3(bucket_name, file_key):
    # Initialize a session using Boto3
    s3_client = boto3.client('s3')

    try:
        # Fetch the JSON file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the content of the file
        json_content = response['Body'].read().decode('utf-8')
        
        # Parse the JSON content
        json_data = json.loads(json_content)
        
        return json_data

    except Exception as e:
        print(f"Error fetching or parsing JSON from S3: {e}")
        return None


def lambda_handler(event, context):
    base_image = get_image_froms3(event)
    output_bucket = event['s3_bucket_name']


    reccomendations_json = load_json_from_s3(output_bucket, "/hyper-personalisation/batchsegmentjob_input.json")

    if reccomendations_json:
        # Process each object in the JSON data
        for item in reccomendations_json:
            item_id = item.get("input", {}).get("itemId")
            users_list = item.get("output", {}).get("usersList", [])

            # Since we only have a few sample images, this id is for a jacket
            # Replace this with "202ee1c4-22af-4329-8672-b218174bf293" for bags or "0c4744e2-b989-4509-a7e2-7d8dc43ff404" for plants
            if item_id == "1de0c711-042b-4b47-93d9-3a7d8d969ac61de0c711-042b-4b47-93d9-3a7d8d969ac6":
                break

            
        print(f"Item ID: {item_id}")
        print(f"Users List: {users_list}")
        print("-" * 40)
    

    

    # create the prompt
    prompt_data = f"""
    
    Human: Generate a prompt for titan image generator to modify an image of "Jackets" to make it more appealing to a young adult auidence limit it to 10 words
    
    Assistant:"""
    image_b64 = image_to_base64(base_image)
    print (image_b64)
    
    outputText = get_completion(prompt_data)
    print(outputText)
    image_uri = call_image_gen_amazon_nova(outputText, image_b64, output_bucket)
    

    json_template = "{subject: , content: }"
    
    email_prompt_data = f"""
    
    
    Human: Generate a HTML email to sell "Jackets" to a young adult auidence and out ouput ONLY a JSON with subject, and content in HTML and leave a space for a  the generated image in base64 as part of the content. 
    
    Do not try to fill the <img src="cid:sample_image"/> tag
    
    Fit the output in the provided JSON template {json_template}. Do not output any \n or newline.

    Here's a sample email HTML that you can use to generate the email content

        <html>
        <body>
            <p>This is a test email with an embedded image:</p>
            <img src="cid:sample_image"/>
        </body>
        </html>
    
    Assistant: 
    Here is the JSON to be outputted: 
    """ 

    email_content = get_completion(email_prompt_data)
    #String replacement only because we are using Base 64. For actual production use a CDN URL.
    #email_content = email_content.replace("<!-- INSERT_IMAGE_HERE -->", image_uri)

    data = json.loads(email_content)
    image_data = new_entry = {
    "image_uri": image_uri
    }
    data["image_uri"] = image_uri
    data_string = json.dumps(data)




    s3_json_uploader(output_bucket,data_string)

    return {
        'statusCode': 200,
        'body': json.dumps('')
    }
