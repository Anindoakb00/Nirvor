import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nirvor Clinical Intake",
  description:
    "Upload voice notes, prescriptions, and scans — view grounded clinical results.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
