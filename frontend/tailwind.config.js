/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        exiton: {
          bg: "#020617",      // 背景
          card: "#020819",    // カード
          accent: "#38bdf8",  // 青
          neon: "#a855f7",    // 紫
        },
      },
      boxShadow: {
        exiton: "0 18px 45px rgba(56, 189, 248, 0.35)",
      },
    },
  },
  plugins: [],
};
