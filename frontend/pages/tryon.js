import { useState } from 'react';
import LiveTryOn from '../components/LiveTryOn';

export default function TryOnPage() {
  const [category, setCategory] = useState('clothing');
  const [selectedProduct, setSelectedProduct] = useState(null);

  const sampleProducts = [
    { id: 1, name: 'Luxury Metallic Watch', category: 'watch', img: '/samples/watch.png' },
    { id: 2, name: 'Casual Black T-Shirt', category: 'clothing', img: '/samples/shirt.png' },
    { id: 3, name: 'Classic Sunglasses', category: 'watch', img: '/samples/glasses.png' }
  ];

  return (
    <div style={{
      minHeight: '100vh',
      width: '100vw',
      backgroundColor: '#f8fafc', // Clean light background instead of bg-store.jpg
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '40px 20px',
      fontFamily: "'Segoe UI', Roboto, sans-serif"
    }}>
      <div style={{
        backgroundColor: '#ffffff',
        padding: '30px 40px',
        borderRadius: '20px',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.08)',
        border: '1px solid #e2e8f0',
        maxWidth: '850px',
        width: '100%',
        marginBottom: '25px',
        textAlign: 'center'
      }}>
        <h1 style={{ fontSize: '30px', fontWeight: '800', color: '#0f172a', marginBottom: '16px' }}>
          AI Product Trial Room
        </h1>

        <div style={{ marginBottom: '20px' }}>
          <p style={{ fontSize: '14px', fontWeight: '700', color: '#334155', marginBottom: '10px' }}>
            Choose a Sample Product to Try On:
          </p>
          <div style={{ display: 'flex', gap: '15px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {sampleProducts.map((item) => (
              <div
                key={item.id}
                onClick={() => {
                  setSelectedProduct(item.img);
                  setCategory(item.category);
                }}
                style={{
                  border: selectedProduct === item.img ? '2px solid #2563eb' : '1px solid #cbd5e1',
                  borderRadius: '12px',
                  padding: '10px 16px',
                  backgroundColor: '#ffffff',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '13px',
                  color: selectedProduct === item.img ? '#2563eb' : '#475569',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.05)',
                  transition: 'all 0.2s'
                }}
              >
                {item.name}
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ textAlign: 'left' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: '700', color: '#334155', marginBottom: '4px' }}>
              Category:
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              style={{
                padding: '8px 14px',
                borderRadius: '8px',
                border: '1px solid #cbd5e1',
                fontSize: '13px',
                outline: 'none',
                backgroundColor: '#ffffff'
              }}
            >
              <option value="clothing">Apparel (Body Fit)</option>
              <option value="watch">Watch / Wrist Accessory</option>
            </select>
          </div>

          <div style={{ textAlign: 'left' }}>
            <label style={{ display: 'block', fontSize: '13px', fontWeight: '700', color: '#334155', marginBottom: '4px' }}>
              Or Upload Custom Image:
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedProduct(e.target.files[0]);
                  setCategory('watch');
                }
              }}
              style={{ fontSize: '13px' }}
            />
          </div>
        </div>
      </div>

      <div style={{
        width: '100%',
        maxWidth: '850px',
        borderRadius: '20px',
        overflow: 'hidden',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        border: '2px solid #e2e8f0'
      }}>
        <LiveTryOn category={category} customImage={selectedProduct} />
      </div>
    </div>
  );
}