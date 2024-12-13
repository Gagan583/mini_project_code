import cv2
import easyocr
import json
import os
import numpy as np
import re
from datetime import datetime
from config import LICENSE_DATA_PATH, UNAUTHORIZED_PATH, UNAUTHORIZED_DATA_PATH

# Initialize EasyOCR reader for license plate recognition globally
reader = easyocr.Reader(["en"], gpu=True)

def apply_effects(image):
    if len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image 
    return gray

# Function for license plate format validation
def validate_license_format(license_text):
    cleaned_text = "".join(filter(str.isalnum, license_text)).upper()

    if re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}", cleaned_text):
        return cleaned_text

    components = re.findall(r"[A-Z]{1,2}|[0-9]{1,4}", cleaned_text)
    if len(components) >= 4:
        state_code = "".join(filter(str.isalpha, components[0]))[:2]
        district_code = "".join(filter(str.isdigit, components[1]))[:2]
        series = "".join(filter(str.isalpha, components[2]))[:2]
        unique_number = "".join(filter(str.isdigit, "".join(components[3:])))[:4]

        if len(state_code) == 2 and len(district_code) == 2 and len(series) == 2 and len(unique_number) == 4:
            return f"{state_code}{district_code}{series}{unique_number}"
    return ""

# Function for extracting license text
def extract_license_text(image):
    preprocessed_image = apply_effects(image)
    results = reader.readtext(preprocessed_image, detail=0, text_threshold=0.6)
    license_text = "".join(results).replace(" ", "").upper()
    return validate_license_format(license_text)

# Function for retrieving license info from the database
def get_license_info(license_number):
    try:
        with open(LICENSE_DATA_PATH, "r") as file:
            license_data = json.load(file)
        
        # Check if the license number exists in the data
        if license_number in license_data:
            return license_data[license_number]
        else:
            return None  # Return None if no match is found
    
    except FileNotFoundError:
        print("Error: License data file not found.")
        return None  

# Function to check authorization status
def check_authorized_plate(license_number):
    # Get the license information for the given plate number
    license_info = get_license_info(license_number)
    if license_info and license_info.get("authorization_status") == "Authorized":
        return True
    return False

# Function to display license info on the frame
def display_license_info(frame, license_number, x1, y1, plate_color):
    info = get_license_info(license_number)

    if info:
        name = info["name"]
        profession = info["profession"]
        auth_status = info["authorization_status"]
        info_text = f"Name: {name}\nProfession: {profession}\nStatus: {auth_status}"
        cv2.putText(frame, f"Plate: {license_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, plate_color, 2)
        cv2.putText(frame, info_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, plate_color, 2)
    else:
        cv2.putText(frame, f"Unknown Plate: {license_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


def save_unauthorized_plates(unauthorized_plate_image, license_number, file_path=UNAUTHORIZED_DATA_PATH):
    try:
        # Ensure the unauthorized folder exists
        if not os.path.exists(UNAUTHORIZED_PATH):
            os.makedirs(UNAUTHORIZED_PATH)

        # Check if the file with the data already exists, if so, load it
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                existing_data = json.load(file)
        else:
            existing_data = []
        
        # Check if the license number already exists in the data
        if any(data["license_number"] == license_number for data in existing_data):
            print(f"License number {license_number} already exists. Skipping save.")
            return  # Exit the function without saving the image or data

        # If the license number does not exist, proceed with saving the image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"unauthorized_{license_number}_{timestamp}.png"
        image_filepath = os.path.join(UNAUTHORIZED_PATH, image_filename)
        
        # Save the license plate image
        cv2.imwrite(image_filepath, unauthorized_plate_image)
        
        # Append the unauthorized plate data to the JSON file
        existing_data.append({
            "license_number": license_number,
            "timestamp": timestamp,
        })

        # Write the updated data back to the file
        with open(file_path, "w") as file:
            json.dump(existing_data, file, indent=4)
        
        print(f"Unauthorized plate saved: {image_filename}")
        
    except Exception as e:
        print(f"Error while saving unauthorized plates: {e}")
