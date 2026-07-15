import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Chat with your PDFs — Senior RAG",
  description: "Hybrid retrieval RAG with streaming answers",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
