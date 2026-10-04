# AI Waste Segregation Using Computer Vision

A web application that uses computer vision to classify waste items into different categories (recyclable, organic, hazardous, non-recyclable).

## Tech Stack

- **Frontend**: React (with Vite)
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow with MobileNetV2 (transfer learning)

## Features

- Upload images of waste items
- AI-powered classification using computer vision
- Real-time confidence scores
- Clean, responsive UI
- Four waste categories: Recyclable, Organic, Hazardous, Non-recyclable

## Project Structure

```.
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # Model training script
│   ├── requirements.txt    # Python dependencies
│   └── waste_model.h5      # Trained model (generated after training)
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── App.css         # Styling
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. (Optional) Train the model with your dataset:
```bash
python train_model.py
```

To train with your own data, organize images as:
```
dataset/
  ├── recyclable/
  ├── organic/
  ├── hazardous/
  └── non-recyclable/
```

6. Start the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

1. Open the web application in your browser
2. Click "Choose Image" to upload an image of waste
3. Click "Classify Waste" to analyze the image
4. View the classification result with confidence score

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/categories` - Get all waste categories
- `POST /api/classify` - Classify an uploaded image

## Model Information

The application uses transfer learning with MobileNetV2 pre-trained on ImageNet. For production use, train the model with a proper dataset of waste images specific to your use case.

## Current Mode

The application currently runs in **simulation mode** if no trained model is found. In this mode, it randomly assigns categories for demonstration purposes. To use real AI classification:

1. Collect images for each waste category
2. Organize them in the `dataset/` folder structure
3. Run `python train_model.py` in the backend directory
4. Restart the Flask server

## Future Enhancements

- Add more waste categories
- Implement real-time camera capture
- Add bulk image processing
- Include waste disposal tips for each category
- Add multilingual support
- Deploy to cloud platforms

## License

MIT License
