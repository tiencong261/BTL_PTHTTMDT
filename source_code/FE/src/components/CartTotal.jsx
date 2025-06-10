import React, { useContext } from "react";
import { ShopContext } from "../context/ShopContext";
import Title from "./Title";

const CartTotal = () => {
  const { currency, delivery_fee, getCartAmount } = useContext(ShopContext);

  return (
    <div className="w-full">
      <div className="text-2xl">
        <Title text1={"TỔNG"} text2={"GIỎ HÀNG"} />
      </div>

      <div className="flex flex-col gap-2 mt-2 text-sm">
        <div className="flex justify-between items-center">
          <p className="text-sm">Subtotal</p>
          <p className="text-sm">
            {getCartAmount()}.00 {currency}
          </p>
        </div>
        <hr />
        <div className="flex justify-between items-center">
          <p className="text-sm">Shipping Fee</p>
          <p className="text-sm">
            {delivery_fee}.00 {currency}
          </p>
        </div>
        <hr />
        <div className="flex justify-between items-center">
          <p className="text-sm font-medium">Total</p>
          <p className="text-sm font-medium">
            {getCartAmount() === 0 ? 0 : getCartAmount() + delivery_fee}.00{" "}
            {currency}
          </p>
        </div>
      </div>
    </div>
  );
};

export default CartTotal;
