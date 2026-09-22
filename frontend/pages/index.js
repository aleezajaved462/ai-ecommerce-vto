import Link from 'next/link';

export default function Home() {
  return (
    <div style={{
      minHeight: '100vh',
      width: '100vw',
      backgroundImage: 'url("/bg-store.jpg")', // Shopping cart background applied here
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      fontFamily: "'Segoe UI', Roboto, sans-serif"
    }}>
      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.85)',
        backdropFilter: 'blur(10px)',
        padding: '50px 60px',
        borderRadius: '24px',
        boxShadow: '0 20px 40px rgba(0,0,0,0.15)',
        textAlign: 'center',
        maxWidth: '600px'
      }}>
        <h1 style={{ fontSize: '38px', fontWeight: '800', color: '#0f172a', marginBottom: '12px' }}>
          AI E-Commerce Virtual Try-On
        </h1>
        <p style={{ fontSize: '18px', color: '#475569', marginBottom: '30px' }}>
          Welcome to the AI-powered store!
        </p>
        <Link href="/tryon">
          <button style={{
            backgroundColor: '#0070f3',
            color: '#ffffff',
            padding: '14px 28px',
            fontSize: '16px',
            fontWeight: '600',
            border: 'none',
            borderRadius: '12px',
            cursor: 'pointer',
            boxShadow: '0 4px 14px rgba(0, 112, 243, 0.39)'
          }}>
            Launch Live Virtual Try-On
          </button>
        </Link>
      </div>
    </div>
  );
}