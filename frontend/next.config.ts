import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
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
