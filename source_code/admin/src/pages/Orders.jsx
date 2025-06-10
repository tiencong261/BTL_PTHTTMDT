import React, { useEffect } from "react";
import { useState } from "react";
import axios from "axios";
import { backendUrl, currency } from "../App";
import { toast } from "react-toastify";
import { assets } from "../assets/assets";
import Modal from "react-modal";

const Orders = ({ token }) => {
  const [orders, setOrders] = useState([]);
  const [showReturnOnly, setShowReturnOnly] = useState(false);
  const [modalOrder, setModalOrder] = useState(null);

  const fetchAllOrders = async () => {
    if (!token) {
      return null;
    }

    try {
      const response = await axios.post(
        backendUrl + "/api/order/list",
        {},
        { headers: { token } }
      );
      // console.log(response.data);
      if (response.data.success) {
        setOrders(response.data.orders);
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  };

  const deleteAllOrders = async () => {
    if (!token) {
      toast.error("Please login first");
      return;
    }

    try {
      console.log("Deleting all orders...");
      const response = await axios.post(
        backendUrl + "/api/order/delete-all",
        {},
        {
          headers: {
            token,
            "Content-Type": "application/json",
          },
        }
      );

      console.log("Delete response:", response.data);

      if (response.data.success) {
        toast.success("All orders have been deleted");
        setOrders([]);
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.error("Error deleting orders:", error);
      toast.error(error.message);
    }
  };

  // const statusHandler = async (event, orderId) => {
  //   try {
  //     const response = await axios.post(backendUrl+"/api/order/status", {orderId, status:event.target.value}, {headers:{token}})
  //     if (response.data.success) {
  //       await fetchAllOrders();
  //     }

  //   } catch (error) {
  //     console.log(error);
  //     toast.error(error.message);
  //   }
  // }

  const statusHandler = async (event, orderId) => {
    try {
      const response = await axios.post(
        backendUrl + "/api/order/status",
        { orderId, status: event.target.value },
        { headers: { token } }
      );

      if (response.data.success) {
        await fetchAllOrders();
        console.log("Status updated successfully:", response.data);
      } else {
        console.error("Failed to update status:", response.data);
        toast.error(response.data.message);
      }
    } catch (error) {
      console.error("Error updating status:", error);
      toast.error(error.message);
    }
  };

  // Đếm số lượng yêu cầu đổi trả/hoàn tiền đang chờ xử lý
  const pendingReturnCount = orders.filter(
    (order) =>
      order.returnRequest &&
      order.returnRequest.requested &&
      order.returnRequest.status === "pending"
  ).length;

  useEffect(() => {
    fetchAllOrders();
  }, [token]);

  return (
    <div>
      <Modal
        isOpen={!!modalOrder}
        onRequestClose={() => setModalOrder(null)}
        ariaHideApp={false}
        style={{ content: { maxWidth: 500, margin: "auto", borderRadius: 8 } }}
      >
        {modalOrder && (
          <div>
            <h2 className="text-lg font-bold mb-2">
              Chi tiết đổi trả/hoàn tiền
            </h2>
            <p>
              <b>Khách hàng:</b> {modalOrder.address.firstName}{" "}
              {modalOrder.address.lastName}
            </p>
            <p>
              <b>Lý do:</b> {modalOrder.returnRequest.reason}
            </p>
            <p>
              <b>Trạng thái:</b> {modalOrder.returnRequest.status}
            </p>
            {modalOrder.returnRequest.adminNote && (
              <p>
                <b>Ghi chú admin:</b> {modalOrder.returnRequest.adminNote}
              </p>
            )}
            <div className="flex gap-2 mt-4">
              {modalOrder.returnRequest.status === "pending" && (
                <>
                  {/* Nếu đơn chưa thanh toán chỉ cho phép duyệt */}
                  {modalOrder.payment === false ? (
                    <button
                      className="px-3 py-1 bg-green-500 text-white rounded text-xs"
                      onClick={async () => {
                        const adminNote = prompt(
                          "Ghi chú khi duyệt đổi trả/hoàn tiền:"
                        );
                        await axios.post(
                          backendUrl + "/api/order/process-return",
                          {
                            orderId: modalOrder._id,
                            status: "approved",
                            adminNote,
                          },
                          { headers: { token } }
                        );
                        localStorage.setItem("revenueNeedsUpdate", Date.now());
                        toast.success("Đã duyệt yêu cầu đổi trả/hoàn tiền!");
                        setModalOrder(null);
                        fetchAllOrders();
                      }}
                    >
                      Duyệt yêu cầu
                    </button>
                  ) : (
                    <>
                      <button
                        className="px-3 py-1 bg-green-500 text-white rounded text-xs"
                        onClick={async () => {
                          const adminNote = prompt(
                            "Ghi chú khi duyệt đổi trả/hoàn tiền:"
                          );
                          await axios.post(
                            backendUrl + "/api/order/process-return",
                            {
                              orderId: modalOrder._id,
                              status: "approved",
                              adminNote,
                            },
                            { headers: { token } }
                          );
                          localStorage.setItem(
                            "revenueNeedsUpdate",
                            Date.now()
                          );
                          toast.success(
                            "Đã duyệt yêu cầu đổi trả/hoàn tiền! Doanh thu đã được cập nhật."
                          );
                          setModalOrder(null);
                          fetchAllOrders();
                        }}
                      >
                        Duyệt yêu cầu
                      </button>
                      <button
                        className="px-3 py-1 bg-red-500 text-white rounded text-xs"
                        onClick={async () => {
                          const adminNote = prompt("Lý do từ chối:");
                          await axios.post(
                            backendUrl + "/api/order/process-return",
                            {
                              orderId: modalOrder._id,
                              status: "rejected",
                              adminNote,
                            },
                            { headers: { token } }
                          );
                          localStorage.setItem(
                            "revenueNeedsUpdate",
                            Date.now()
                          );
                          toast.success(
                            "Đã từ chối yêu cầu đổi trả/hoàn tiền!"
                          );
                          setModalOrder(null);
                          fetchAllOrders();
                        }}
                      >
                        Từ chối yêu cầu
                      </button>
                    </>
                  )}
                  <button
                    className="px-3 py-1 bg-gray-300 rounded text-xs"
                    onClick={() => setModalOrder(null)}
                  >
                    Đóng
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </Modal>
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-4">
          <h3 className="text-xl font-bold">Trang Đơn Hàng</h3>
          {/* Badge số lượng yêu cầu đổi trả */}
          {pendingReturnCount > 0 && (
            <span className="bg-yellow-400 text-white px-2 py-1 rounded text-xs font-semibold">
              {pendingReturnCount} yêu cầu đổi trả/hoàn tiền mới
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowReturnOnly((v) => !v)}
            className={`px-4 py-2 rounded ${
              showReturnOnly
                ? "bg-yellow-500 text-white"
                : "bg-gray-200 text-gray-700"
            } font-semibold`}
          >
            {showReturnOnly ? "Xem tất cả đơn" : "Chỉ xem đơn đổi trả"}
          </button>
          <button
            onClick={deleteAllOrders}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Xóa Tất Cả Đơn Hàng
          </button>
        </div>
      </div>
      <div>
        {/* Header row */}
        <div className="hidden lg:grid grid-cols-1 sm:grid-cols-[0.5fr_2fr_1fr] lg:grid-cols-[0.5fr_2fr_1fr_1fr_1fr] gap-3 items-center border-b-2 border-gray-300 px-5 py-2 font-semibold bg-gray-100 text-gray-700 text-xs sm:text-sm mb-2">
          <div>Sản phẩm</div>
          <div>Thông tin giao hàng</div>
          <div>Thông tin đơn hàng</div>
          <div>Tổng tiền</div>
          <div>Trạng thái</div>
        </div>
        {(showReturnOnly
          ? orders.filter(
              (order) => order.returnRequest && order.returnRequest.requested
            )
          : orders
        ).map((order, index) => (
          <div
            className="grid grid-cols-1 sm:grid-cols-[0.5fr_2fr_1fr] lg:grid-cols-[0.5fr_2fr_1fr_1fr_1fr] gap-3 items-start border-2 border-gray-200 p-5 md:p-8 my-3 md:my-4 text-xs sm:text-sm text-gray-700 cursor-pointer"
            key={index}
            onClick={() =>
              order.returnRequest && order.returnRequest.requested
                ? setModalOrder(order)
                : null
            }
          >
            <div className="flex flex-col gap-2">
              {order.items.map((item, index) => (
                <div key={index} className="flex items-center gap-2">
                  <img
                    src={
                      item.image?.[0] ||
                      "https://via.placeholder.com/100x100?text=No+Image"
                    }
                    alt={item.name}
                    className="w-12 h-12 object-cover rounded"
                    onError={(e) => {
                      if (
                        e.target.src !==
                        "https://via.placeholder.com/100x100?text=No+Image"
                      ) {
                        e.target.src =
                          "https://via.placeholder.com/100x100?text=No+Image";
                      }
                    }}
                  />
                  <div>
                    <p className="font-medium">Tên: {item.name}</p>
                    <p className="text-gray-500">Size: {item.size}</p>
                    <p className="text-gray-500">Số lượng: {item.quantity}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="pl-4">
              <p className="mt-3 mb-2 font-medium">
                Tên khách hàng:{" "}
                {order.address.firstName + " " + order.address.lastName}
              </p>
              <div>
                <p>Địa chỉ: {order.address.street},</p>
                <p>
                  {order.address.city +
                    ", " +
                    order.address.state +
                    ", " +
                    order.address.country +
                    ", " +
                    order.address.zipcode}
                </p>
              </div>
              <p className="mt-2">Số điện thoại: {order.address.phone}</p>
            </div>
            <div>
              <p className="text-sm sm:text-[15px]">
                Số lượng: {order.items.length}
              </p>
              <p className="mt-3">Phương thức: {order.paymentMethod}</p>
              <p>
                Thanh toán: {order.payment ? "Đã thanh toán" : "Chờ thanh toán"}
              </p>
              <p>Ngày đặt: {new Date(order.date).toLocaleDateString()}</p>
              {/* Hiển thị trạng thái đổi trả/hoàn tiền nếu có */}
              {order.returnRequest && order.returnRequest.requested && (
                <div className="mt-2 p-2 bg-yellow-50 border border-yellow-300 rounded">
                  <p className="text-yellow-700 font-semibold">
                    Yêu cầu đổi trả/hoàn tiền:{" "}
                    <b>{order.returnRequest.status}</b>
                  </p>
                  <p className="text-xs text-gray-700">
                    Lý do: {order.returnRequest.reason}
                  </p>
                  {order.returnRequest.adminNote && (
                    <p className="text-xs text-gray-500">
                      Ghi chú admin: {order.returnRequest.adminNote}
                    </p>
                  )}
                  {/* Nếu trạng thái là pending, cho phép admin xử lý */}
                  {order.returnRequest.status === "pending" && (
                    <div className="mt-2 flex flex-col gap-1">
                      <button
                        className="px-2 py-1 bg-green-500 text-white rounded text-xs"
                        onClick={async () => {
                          const adminNote = prompt(
                            "Ghi chú khi duyệt đổi trả/hoàn tiền:"
                          );
                          await axios.post(
                            backendUrl + "/api/order/process-return",
                            {
                              orderId: order._id,
                              status: "approved",
                              adminNote,
                            },
                            { headers: { token } }
                          );
                          localStorage.setItem(
                            "revenueNeedsUpdate",
                            Date.now()
                          );
                          toast.success("Đã duyệt yêu cầu đổi trả/hoàn tiền!");
                          fetchAllOrders();
                        }}
                      >
                        Duyệt yêu cầu
                      </button>
                      <button
                        className="px-2 py-1 bg-red-500 text-white rounded text-xs"
                        onClick={async () => {
                          const adminNote = prompt("Lý do từ chối:");
                          await axios.post(
                            backendUrl + "/api/order/process-return",
                            {
                              orderId: order._id,
                              status: "rejected",
                              adminNote,
                            },
                            { headers: { token } }
                          );
                          localStorage.setItem(
                            "revenueNeedsUpdate",
                            Date.now()
                          );
                          toast.success(
                            "Đã từ chối yêu cầu đổi trả/hoàn tiền!"
                          );
                          fetchAllOrders();
                        }}
                      >
                        Từ chối yêu cầu
                      </button>
                      <button
                        className="px-2 py-1 bg-blue-500 text-white rounded text-xs"
                        onClick={async () => {
                          const adminNote = prompt("Ghi chú khi hoàn tiền:");
                          await axios.post(
                            backendUrl + "/api/order/process-return",
                            {
                              orderId: order._id,
                              status: "refunded",
                              adminNote,
                            },
                            { headers: { token } }
                          );
                          toast.success("Đã hoàn tiền cho đơn hàng!");
                          fetchAllOrders();
                        }}
                      >
                        Đánh dấu đã hoàn tiền
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
            <p className="text-sm sm:text-[15px] font-bold">
              {order.amount} {currency}
            </p>
            <select
              onChange={(event) => statusHandler(event, order._id)}
              value={order.status}
              className="p-2 font-semibold border rounded"
            >
              <option value="Order Placed">Đã đặt hàng</option>
              <option value="Packing">Đang đóng gói</option>
              <option value="Shipped">Đã gửi hàng</option>
              <option value="Out for delivery">Đang giao hàng</option>
              <option value="Delivered">Đã giao hàng</option>
            </select>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Orders;

// 12:12
