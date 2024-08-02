import React from "react";
import ReactDOM from "react-dom/client";
import Logo from "./Logo";
import "./styles.css";

const App = () => {
  return (
    <>
      <Logo />
      <button onClick={download} style={{ marginTop: 50 }}>
        Download
      </button>
    </>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

export const download = () => {
  const clone = document.querySelector("svg").cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const data = clone.outerHTML;
  const blob = new Blob([data], { type: "image/svg+xml" });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "logo.svg";
  link.click();
  window.URL.revokeObjectURL(url);
};
