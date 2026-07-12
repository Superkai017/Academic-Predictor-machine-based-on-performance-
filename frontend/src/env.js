// env.js — frontend environment config.
//
// The browser can't read .env files (no bundler in this project), so this
// plain JS file plays that role. It must be loaded BEFORE the main script
// in index.html.
//
// DEPLOYMENT: change API_BASE to your deployed backend URL, e.g.:
//   API_BASE: "https://academic-predictor.onrender.com"
// No other file needs to change.
window.ENV = {
  API_BASE: "http://localhost:5000",
};
