import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Understanding AI and Machine Learning - Interactive eBook',
  description: 'A comprehensive 15-page guide to artificial intelligence and machine learning fundamentals',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
