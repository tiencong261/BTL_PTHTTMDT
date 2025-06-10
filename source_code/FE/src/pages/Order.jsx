import React, { useContext, useEffect, useState } from "react";
import { ShopContext } from "../context/ShopContext";
import Title from "../components/Title";
import axios from "axios";

const Order = () => {
  const { backendUrl, token, currency } = useContext(ShopContext);

  const [orderData, setOrderData] = useState([]);

  const loadOrderData = async () => {
    try {
      if (!token) {
        return null;
      }

      const response = await axios.post(
        backendUrl + "/api/order/userorders",
        {},
        { headers: { token } }
      );

      if (response.data.success) {
        let allOrdersItem = [];
        response.data.orders.map((order) => {
          order.items.map((item) => {
            item["status"] = order.status;
            item["payment"] = order.payment;
            item["paymentMethod"] = order.paymentMethod;
            item["date"] = order.date;
            item["orderId"] = order._id;
            item["returnRequest"] = order.returnRequest;
            allOrdersItem.push(item);
          });
        });
        setOrderData(allOrdersItem.reverse());
      }
    } catch (error) {}
  };

  useEffect(() => {
    loadOrderData();
  }, [token]);

  return (
    <div className="border-t pt-16">
      <div className="text-2xl">
        <Title text1={"ĐƠN HÀNG"} text2={"CỦA TÔI"} />
      </div>

      <div>
        {orderData.map((item, index) => {
          const returnRequest = item.returnRequest;
          return (
            <div
              key={index}
              className="py-4 border-t border-b text-gray-700 flex flex-col md:flex-row md:items-center md:justify-between gap-4"
            >
              <div className="flex items-start gap-6 text-sm">
                <img className="w-16 sm:w-20" src={item.image[0]} alt="" />
                <div>
                  <div className="flex justify-between items-center">
                    <p className="text-sm">{item.name}</p>
                    <p className="text-sm">
                      {item.price} {currency}
                    </p>
                  </div>
                  <div className="flex items-center gap-3 mt-2 text-base text-gray-700">
                    <p>Số lượng: {item.quantity}</p>
                    <p>Size: {item.size}</p>
                  </div>
                  <p className="mt-2">
                    Ngày:{" "}
                    <span className="text-gray-400">
                      {new Date(item.date).toDateString()}
                    </span>
                  </p>
                  <p className="mt-2">
                    Thanh toán:{" "}
                    <span className="text-gray-400">{item.paymentMethod}</span>
                  </p>
                  {returnRequest && returnRequest.requested ? (
                    <p className="mt-2 text-blue-600">
                      Yêu cầu đổi trả: <b>{returnRequest.status}</b>
                      {returnRequest.reason ? ` - ${returnRequest.reason}` : ""}
                    </p>
                  ) : null}
                </div>
              </div>
              <div className="md:w-1/2 flex flex-col gap-2 justify-between">
                <div className="flex items-center gap-2">
                  <p className="min-w-2 h-2 rounded-full bg-green-500"></p>
                  <p className="text-sm md:text-base">{item.status}</p>
                </div>
                <button
                  onClick={loadOrderData}
                  className="border px-4 py-2 text-sm font-medium rounded-sm"
                >
                  Theo dõi đơn hàng
                </button>
                {(!returnRequest || !returnRequest.requested) && (
                  <button
                    className="border px-4 py-2 text-sm font-medium rounded-sm mt-2 bg-yellow-100 hover:bg-yellow-200"
                    onClick={async () => {
                      const reason = prompt("Nhập lý do đổi trả/hoàn tiền:");
                      if (!reason) return;
                      try {
                        await axios.post(
                          backendUrl + "/api/order/return-request",
                          { orderId: item.orderId, reason },
                          { headers: { token } }
                        );
                        alert("Đã gửi yêu cầu đổi trả/hoàn tiền!");
                        loadOrderData();
                      } catch (err) {
                        alert("Gửi yêu cầu thất bại!");
                      }
                    }}
                  >
                    Yêu cầu đổi trả/hoàn tiền
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Order;
