import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'ResearcherAI - Autonomous Deep Research Platform',
  description:
    'A polished website for ResearcherAI, an agentic research platform for citation-grounded answers across private knowledge, live web sources, and developer documentation.',
  openGraph: {
    title: 'ResearcherAI - Autonomous Deep Research Platform',
    description:
      'Route research across documents, web sources, and technical docs, then synthesize verified answers with citations.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ResearcherAI - Autonomous Deep Research Platform',
    description:
      'Route research across documents, web sources, and technical docs, then synthesize verified answers with citations.',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
