import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pressure / Play — Cricket Performance",
  description: "Context adjusted IPL and T20I player pressure rankings.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
