/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        headline: ['"Playfair Display"', 'Georgia', 'serif'],
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
      },
      colors: {
        paper: '#f9f6f0',
        ink: '#1a1a1a',
        rule: '#c9b99a',
        'ink-light': '#4a4a4a',
        masthead: '#1a3a5c',
      },
    },
  },
  plugins: [],
}
