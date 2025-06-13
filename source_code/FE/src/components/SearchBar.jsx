import React, { useContext, useEffect, useState } from 'react'
import { ShopContext } from '../context/ShopContext'
import { assets } from '../assets/assets';
import { useLocation } from 'react-router-dom';

const SearchBar = () => {

    const {search, setSearch, showSearch, setShowSearch} = useContext(ShopContext);
    const location = useLocation();
    const [visible, setVisible] = useState(false);

    useEffect(()=>{
        if(location.pathname.includes('collection') && showSearch){
            setVisible(true);
        }
        else{
            setVisible(false);
        }
    },[location])

    const handleVoiceSearch = () => {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'vi-VN';
        recognition.continuous = false;
        recognition.interimResults = false;
    
        recognition.onstart = () => {
            console.log("Đang bắt đầu nhận diện giọng nói...");
        };
    
        recognition.onresult = async (event) => {
            recognition.stop();  // Dừng ngay khi có kết quả
            const speechText = event.results[0][0].transcript.trim();
            console.log("Kết quả giọng nói:", speechText);
    
            if (speechText !== "") {
                setSearch(speechText);  // Cập nhật luôn bằng kết quả gốc
                
                try {
                    const translatedText = await translateText(speechText);
                    if (translatedText && translatedText !== speechText) {
                        setSearch(translatedText);
                    }
                } catch (err) {
                    console.error("Lỗi dịch:", err);
                }
            }
        };
    
        recognition.onerror = (event) => {
            recognition.stop();  // Dừng nếu lỗi
            if (event.error === 'no-speech') {
                console.log("Không có giọng nói nào được phát hiện.");
            } else {
                console.error("Lỗi giọng nói:", event.error);
            }
        };
    
        recognition.start();
    };

  return showSearch && visible ? (
    <div className='border border-b bg-gray-50 text-center'>
        <div className='inline-flex items-center justify-center border border-gray-400 px-5 py-2 my-5 mx-3 rounded-full w-3/4 sm:w-1/2'>
            <input value={search} onChange={(e) => setSearch(e.target.value)} className='flex-1 outline-none bg-inherit text-sm' type='text' placeholder='Search' />
            <img src={assets.search_icon} alt=''/>
            <button
                onClick={handleVoiceSearch}
                className="ml-2 flex items-center justify-center w-10 h-10 rounded-full border border-gray-400 bg-white hover:bg-[#fff7e6] transition shadow-sm hover:shadow-md"
                aria-label="Voice Search"
                >
                {/* Previous mic icon — trimmed to remove the bottom base */}
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-5 h-5 text-[#3C3C3C] group-hover:text-[#2a2a2a] transition"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3z" />
                    <path d="M17 11a5 5 0 0 1-10 0H5a7 7 0 0 0 14 0h-2z" />
                </svg>
            </button>
        </div>
        <img onClick={()=>setShowSearch(false)} className='inline w-3 cursor-pointer' src={assets.cross_icon} alt=''/>
      
    </div>
  ) : null
}

export default SearchBar
