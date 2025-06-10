import React from "react";
import { assets } from "../assets/assets";
import Title from '../components/Title'
import NewsletterBox from "../components/NewsletterBox";

const Contact = () => {
  return (
    <div>

            <div className="text-2xl text-center pt-10 border-t">
                <Title text1={'LIÊN HỆ'} text2={'VỚI CHÚNG TÔI'}/>
            </div>  
     
            <div className="my-10 flex flex-col md:flex-row gap-10 mb-28 justify-center">
                <img className="w-full md:max-w-[480px]" src={assets.contact_img} alt=""/>
                <div className="flex flex-col justify-center gap-6 md:w-2/4 text-gray-600">
                    <p className="font-semibold text-xl text-gray-600">Cửa hàng của chúng tôi</p>
                    <p className="text-gray-500">Km10, Nguyen Trai, Ha Dong, Ha Noi</p>
                    <p className="text-gray-500">Số điện thoại: 0123456789 <br/> Email:nhom8@gmail.com</p>
                    <p className="font-semibold text-xl text-gray-600">Tuyển dụng tại Forever</p>
                    <p className="text-gray-500">Tìm hiểu thêm về các đội ngũ và cơ hội việc làm của chúng tôi.</p>
                    <button className="border border-black px-8 py-4 text-sm hover:bg-black hover:text-white transition-all duration-50 w-1/4">Khám phá công việc</button>
                </div>
            </div>
           <NewsletterBox />
        </div>
  );
};

export default Contact;

