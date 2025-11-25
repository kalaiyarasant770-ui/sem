const CONFIG = {
  development: {
    ENV: "development",
    BASE_URL: "http://127.0.0.1:5000",
    STATIC_PATH: "/static", // local Flask path
  },
  production: {
    ENV: "production",
    BASE_URL: "https://app.devsbot.com/chatbot/api",
    STATIC_PATH: "/chatbot/static", // live Nginx path
  },
};

// const ACTIVE_CONFIG = CONFIG.production;
const ACTIVE_CONFIG = CONFIG.development;
console.log(ACTIVE_CONFIG);