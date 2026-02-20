/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'cdn.example.com',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'storage.example.com',
        pathname: '/uploads/**',
      },
    ],
  },
  experimental: {
    typedRoutes: true,
  },
}

export default nextConfig
