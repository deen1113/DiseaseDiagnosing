import requests
import argparse

def predicter(image_path):
    url = 'http://127.0.0.1:5000/predict'
    try:
        with open(image_path, 'rb') as file:
            files = {'image': file}
            response = requests.post(url, files=files)
            response.raise_for_status()  
            print("Prediction:", response.json())
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test X-ray classification API.')
    parser.add_argument('image_path', type=str, help='Path to the X-ray image file')
    args = parser.parse_args()

    predicter(args.image_path)