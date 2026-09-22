const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Mock Product Database
const products = [
  { id: '1', name: 'AI Smart Chronograph Watch', category: 'watch', price: 299, image: 'https://via.placeholder.com/200?text=Watch' },
  { id: '2', name: 'Cyberpunk Animated Jacket', category: 'clothing', price: 149, image: 'https://via.placeholder.com/200?text=Jacket' }
];

// Fetch Catalog
app.get('/api/products', (req, res) => {
  res.json({ success: true, data: products });
});

// Route for AI Virtual Try-On
app.post('/api/try-on', async (req, res) => {
  try {
    const { userImageBase64, productType, productImageUrL } = req.body;
    
    // Call Python FastAPI AI Engine
    const aiResponse = await axios.post('http://localhost:8000/api/v1/try-on', {
      user_image_base64: userImageBase64,
      product_type: productType,
      product_image_url: productImageUrL
    });

    res.json(aiResponse.data);
  } catch (error) {
    res.status(500).json({ success: false, message: "AI Processing Failed", error: error.message });
  }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Backend Server running on port ${PORT}`));