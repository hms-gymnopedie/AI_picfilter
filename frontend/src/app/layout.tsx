import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { cn } from '@/lib/utils'
import { ThemeProvider } from '@/components/ui/ThemeProvider'
import { Header } from '@/components/ui/Header'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-geist-sans',
  display: 'swap',
})

export const metadata: Metadata = {
  title: {
    default: 'AI_picfilter',
    template: '%s | AI_picfilter',
  },
  description: 'AI-powered image style learning and filter generation',
  keywords: ['LUT', 'filter', 'style transfer', 'AI', 'photo editing', 'NILUT'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <body className={cn(inter.variable, 'min-h-screen flex flex-col')}>
        <ThemeProvider>
          <Header />
          <main className="flex-1 container mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>
          <footer className="border-t border-gray-200 dark:border-gray-800 py-4 text-center text-xs text-gray-400 dark:text-gray-600">
            AI_picfilter &copy; {new Date().getFullYear()}
          </footer>
        </ThemeProvider>
      </body>
    </html>
  )
}
