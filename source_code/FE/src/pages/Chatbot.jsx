import React, { useState, useEffect, useRef } from "react";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);
  const endOfMessagesRef = useRef(null);

  const sendMessage = async () => {
    if (!userInput.trim() || isLoading) return;

    const userMessage = { sender: "user", text: userInput };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    const inputToSend = userInput;
    setUserInput("");

    try {
      const response = await fetch("http://127.0.0.1:5001/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: inputToSend }),
      });

      const data = await response.json();
      const reply = data.reply || "Xin lỗi, hệ thống đang bảo trì";

      const botMessage = {
        sender: "bot",
        text: reply,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Lỗi gửi yêu cầu:", error);
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !isLoading) sendMessage();
  };

  const formatText = (text) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  };

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Auto scroll lên đầu trang Chatbot khi vào trang
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  // Auto scroll lên đầu trang Chatbot mỗi khi có tin nhắn mới
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [messages]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 flex flex-col items-center py-8 px-4">
      <div className="w-full max-w-4xl">
        <h2 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600 mb-8 text-center flex items-center gap-2 justify-center">
          <span role="img" aria-label="chat">
            💬
          </span>{" "}
          Chatbot AI Bán Hàng
        </h2>
        <div
          className="w-full flex flex-col items-center"
          style={{ minHeight: 500 }}
        >
          <div
            className="w-full flex flex-col flex-grow"
            style={{ minHeight: 400, maxHeight: 500 }}
          >
            <div
              ref={chatContainerRef}
              className="flex-1 w-full bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-6 overflow-y-auto flex flex-col min-h-[300px] max-h-[400px] scrollbar-thin scrollbar-thumb-blue-400 scrollbar-track-blue-100"
            >
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-[250px] text-gray-500">
                  <div className="text-6xl mb-4">
                    <span role="img" aria-label="bot">
                      🤖
                    </span>
                  </div>
                  <p className="text-xl font-medium">
                    Xin chào! Tôi có thể giúp gì cho bạn?
                  </p>
                </div>
              )}
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`mb-4 animate-fadeIn ${
                    msg.sender === "user"
                      ? "flex justify-end"
                      : "flex justify-start"
                  }`}
                >
                  <div
                    className={`px-5 py-3 rounded-2xl text-base sm:text-lg leading-relaxed max-w-[85%] break-words shadow-md ${
                      msg.sender === "user"
                        ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span className="font-bold">
                        {msg.sender === "user" ? "👤 Bạn:" : "🤖 Bot:"}
                      </span>
                      <div
                        className="inline-block"
                        dangerouslySetInnerHTML={{
                          __html: formatText(msg.text),
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="px-5 py-3 rounded-2xl bg-gray-100 text-gray-800 shadow-md">
                    <div className="flex items-center gap-2">
                      <span className="font-bold">🤖 Bot:</span>
                      <div className="flex gap-1">
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0ms" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "150ms" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "300ms" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={endOfMessagesRef} />
            </div>
          </div>
          <div className="w-full mt-4 flex items-center gap-3">
            <input
              ref={inputRef}
              type="text"
              placeholder="Nhập câu hỏi của bạn..."
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={isLoading}
              className="flex-1 px-5 py-3 text-lg border border-blue-300 rounded-full shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all disabled:opacity-50 bg-white"
            />
            <button
              onClick={sendMessage}
              disabled={isLoading}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold text-lg rounded-full shadow-md hover:from-blue-600 hover:to-blue-700 transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <span role="img" aria-label="send">
                🚀
              </span>{" "}
              Gửi
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
