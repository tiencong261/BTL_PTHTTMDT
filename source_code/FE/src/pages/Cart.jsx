import React, { useContext, useEffect, useState } from "react";
import { ShopContext } from "../context/ShopContext";
import Title from "../components/Title";
import { assets } from "../assets/assets";
import CartTotal from "../components/CartTotal";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Cart = () => {
  const { products, currency, cartItems, updateQuantity, navigate } =
    useContext(ShopContext);
  const [cartData, setCartData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);

    const updateCart = () => {
      if (products && products.length > 0 && cartItems) {
        const tempData = [];

        for (const productId in cartItems) {
          const sizes = cartItems[productId];

          for (const size in sizes) {
            if (sizes[size] > 0) {
              const product = products.find((p) => p._id === productId);
              if (product) {
                tempData.push({
                  _id: productId,
                  size: size,
                  quantity: sizes[size],
                });
              }
            }
          }
        }

        setCartData(tempData);
      }
      setLoading(false);
    };

    if (products && products.length > 0) {
      updateCart();
    } else {
      setCartData([]);
      setLoading(false);
    }
  }, [cartItems, products]);

  return (
    <div className="border-t pt-14">
      <div className="text-2xl mb-3">
        <Title text1={"GIỎ HÀNG"} text2={"CỦA TÔI"} />
      </div>

      {loading ? (
        <div className="text-center py-10">Đang tải giỏ hàng...</div>
      ) : cartData.length > 0 ? (
        <>
          <div>
            {cartData.map((item, index) => {
              const productData = products.find(
                (product) => product._id === item._id
              );

              if (!productData) return null;

              return (
                <div
                  key={index}
                  className="py-4 border-t border-b text-gray-700 grid grid-cols-[4fr_0.5fr_0.5fr] sm:grid-cols-[4fr_2fr_0.5fr] items-center gap-4"
                >
                  <div className="flex items-start gap-6">
                    <img
                      className="w-16 sm:w-20"
                      src={productData.image[0]}
                      alt=""
                    />
                    <div>
                      <div className="flex justify-between items-center">
                        <p className="text-sm">{productData.name}</p>
                        <p className="text-sm">
                          {productData.price} {currency}
                        </p>
                      </div>
                      <div className="flex items-center gap-5 mt-2">
                        <p className="px-2 sm:px-3 sm:py-1 border bg-slate-50">
                          {item.size}
                        </p>
                      </div>
                    </div>
                  </div>
                  <input
                    onChange={(e) =>
                      e.target.value === "" || e.target.value === "0"
                        ? null
                        : updateQuantity(
                            item._id,
                            item.size,
                            Number(e.target.value)
                          )
                    }
                    className="border max-w-10 sm:max-w-20 px-1 sm:px-2 py-1"
                    type="number"
                    min={1}
                    defaultValue={item.quantity}
                  />
                  <img
                    onClick={() => updateQuantity(item._id, item.size, 0)}
                    className="w-4 mr-4 sm:w-5 cursor-pointer"
                    src={assets.bin_icon}
                    alt=""
                  />
                </div>
              );
            })}
          </div>

          <div className="flex justify-end my-20">
            <div className="w-full sm:w-[450px]">
              <CartTotal />
              <div className="w-full text-end">
                <button
                  onClick={() => navigate("/place-order")}
                  className="bg-black text-white text-sm my-8 px-8 py-3"
                >
                  PROCEED TO CHECKOUT
                </button>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-10">
          <p>Giỏ hàng của bạn trống</p>
          <button
            onClick={() => navigate("/")}
            className="bg-black text-white text-sm mt-4 px-8 py-3"
          >
            TIẾP TỤC MUA SẮM
          </button>
        </div>
      )}
    </div>
  );
};

export default Cart;
