---

# Pneumonia Detection Backend

This is a FastAPI-based backend service designed for detecting **pneumonia** from chest X-ray images using a **MobileNet model** trained on a dataset of X-ray images. The model classifies the X-ray images as either **normal** or showing **pneumonia**.

The backend provides an API endpoint that allows users to upload an image and receive a prediction based on the model's inference.

---

## Features

* **FastAPI** backend framework for efficient API handling.
* **TensorFlow** model for pneumonia detection.
* **Image preprocessing** to resize and normalize the input before prediction.
* **Logging** to track predictions and errors.
* **CORS middleware** to enable frontend and backend communication across domains.

---

## How It Works

1. **Upload Image**: The user sends an image (X-ray) through the `/predict` endpoint.
2. **Image Processing**: The image is preprocessed by resizing it to the required input shape, converting it to a suitable format for the model, and normalizing pixel values.
3. **Prediction**: The preprocessed image is passed to a **MobileNet** model to predict whether the image depicts a normal chest or pneumonia.
4. **Response**: The API returns a prediction along with the confidence score.

---

## Setup and Installation

### Prerequisites

Before setting up the project, make sure you have the following installed:

* **Python 3.7+**
* **TensorFlow** (for the model)
* **FastAPI**
* **Uvicorn** (for running the FastAPI server)
* **Pillow** (for image handling)

### 1. Clone the Repository

```bash
git clone https://github.com/Evertte/pneumonia-detection-backend.git
cd pneumonia-detection-backend
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

Create a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### 3. Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

Make sure you have the following dependencies in `requirements.txt`:

```plaintext
fastapi
uvicorn
tensorflow
pillow
numpy
```

### 4. Running the App Locally

To start the server locally, use the following command:

```bash
uvicorn app:app --reload
```

The backend will be running at `http://localhost:8000`. You can test the API using tools like **Postman** or **curl**.

---

## API Documentation

Once the app is running, you can access the **Swagger UI** for API documentation by visiting `http://localhost:8000/docs`.

### Endpoints

#### `POST /predict`

Uploads an X-ray image and predicts if it shows **pneumonia** or is **normal**.

##### Request:

* **Content-Type**: `multipart/form-data`
* **Parameters**:

  * `file`: X-ray image (JPEG/PNG format)

##### Response:

```json
{
  "prediction": "PNEUMONIA", 
  "confidence": 0.89
}
```

* **prediction**: The model's prediction (`PNEUMONIA` or `NORMAL`).
* **confidence**: The confidence score for the prediction (between 0 and 1).

---

## Deployment

This backend is ready to be deployed on platforms like **Render** or **Heroku**. Follow the steps below to deploy it:

### Render Deployment:

1. Push your code to **GitHub**.
2. Link your GitHub repository to **Render**.
3. Configure **Render** to deploy the app using the `render.yaml` file.

For detailed steps, check out the [Render documentation](https://render.com/docs).

---

## Notes

* **Model File**: The model is trained and stored in the `mobilenet_model.h5` file. Make sure it is available during deployment.
* **Image Preprocessing**: The images are resized to 224x224 pixels and normalized before feeding them to the model.
* **CORS**: The app allows all origins for development purposes. In production, replace the wildcard with your frontend URL.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* **TensorFlow**: For providing the pre-trained models and making the model training easier.
* **FastAPI**: For providing a fast and modern web framework for building APIs.
* **MobileNet**: The model used for this pneumonia detection task.

---

### 🎉 **Contributions** are welcome! Feel free to fork the repo, raise issues, or submit pull requests.

---

### Customizations

* Feel free to adjust or add any specific setup steps that might apply to your environment (e.g., cloud storage for the model, environment variables for sensitive data, etc.).

---
