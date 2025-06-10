import React from "react";
import { assets } from "../assets/assets";

const Footer = () => {
  return (
    <div>
      <div className="flex flex-col sm:grid grid-cols-[3fr_1fr_1fr] gap-14 my-10 mt-40 text-sm ">
        <div>
          <img src={assets.logo} className="mb-5 w-32" alt="" />
          <p className="w-full md:w-2/3 text-gray-600">
            Chúng tôi mang đến cho bạn những thiết kế thời trang hiện đại, phong
            cách và chất lượng. Tất cả các sản phẩm đều được chọn lọc kỹ lưỡng
            để giúp bạn tự tin thể hiện cá tính.
          </p>
        </div>

        <div>
          <p className="text-xl font-medium mb-5">CÔNG TY</p>
          <ul className="flex flex-col gap-1 text-gray-600">
            <li>Trang chủ</li>
            <li>Giới thiệu</li>
            <li>Vận chuyển</li>
            <li>Chính sách bảo mật</li>
          </ul>
        </div>

        <div>
          <p className="text-xl font-medium mb-5">Liên hệ</p>
          <ul className="flex flex-col gap-1 text-gray-600">
            <li>0123456789</li>
            <li>nhom8@ecommerce.com</li>
          </ul>
        </div>
      </div>

      <div className="">
        <hr />
        <p className="py-5 text-sm text-center">
          Copyright 2025@ecommerce.com - All Right Reserved.
        </p>
      </div>
    </div>
  );
};

export default Footer;
