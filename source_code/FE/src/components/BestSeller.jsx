import React, { useContext, useEffect, useState } from "react";
import { ShopContext } from "../context/ShopContext";
import ProductItem from "./ProductItem";
import Title from "./Title";

const BestSeller = () => {
  const { products } = useContext(ShopContext);
  const [bestSeller, setBetSeller] = useState([]);

  useEffect(() => {
    const bestProduct = products.filter((item) => item.bestseller);
    setBetSeller(bestProduct.slice(0, 5));
  }, [products]);

  return (
    <div className="my-10">
      <div className="text-center text-3xl py-8">
        <Title text1={"SẢN PHẨM"} text2={"BÁN CHẠY"} />
        <p className="w-3/4 m-auto text-xs sm:text-sm md:text-base text-gray-600">
          SẢN PHẨM ĐƯỢC YÊU THÍCH VÀ BÁN CHẠY NHẤT
        </p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 gap-y-6">
        {bestSeller.map((item, index) => (
          <ProductItem
            key={index}
            id={item.id}
            name={item.name}
            image={item.image}
            price={item.price * 1000}
          />
        ))}
      </div>
    </div>
  );
};

export default BestSeller;
