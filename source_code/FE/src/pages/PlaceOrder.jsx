import React, { useContext, useState } from "react";
import Title from "../components/Title";
import { assets } from "../assets/assets";
import CartTotal from "../components/CartTotal";
import { ShopContext } from "../context/ShopContext";
import axios from "axios";
import { toast } from "react-toastify";
import SolanaPayment from "../components/SolanaPayment";

const PlaceOrder = () => {
  const [method, setMethod] = useState("cod");
  const [showSolanaPayment, setShowSolanaPayment] = useState(false);
  const [solanaPaymentData, setSolanaPaymentData] = useState(null);
  const [currentOrderId, setCurrentOrderId] = useState(null);
  const {
    navigate,
    backendUrl,
    token,
    cartItems,
    setCartItems,
    getCartAmount,
    delivery_fee,
    products,
  } = useContext(ShopContext);
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    street: "",
    city: "",
    state: "",
    zipcode: "",
    country: "",
    phone: "",
  });

  const onChangeHandle = (event) => {
    const { name, value } = event.target;
    setFormData((data) => ({ ...data, [name]: value }));
  };

  const initPay = (order) => {
    const options = {
      key: import.meta.env.VITE_RAZORPAY_KEY_ID,
      amount: order.amount,
      currency: order.currency,
      name: "Order Payment",
      description: "Order Payment",
      order_id: order.id,
      receipt: order.receipt,
      handler: async (response) => {
        try {
          const verifyResponse = await axios.post(
            backendUrl + "/api/order/verifyRazorpay",
            {
              orderId: order.id,
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_order_id: response.razorpay_order_id,
              razorpay_signature: response.razorpay_signature,
            },
            { headers: { token } }
          );
          if (verifyResponse.data.success) {
            setCartItems({});
            navigate("/orders");
          } else {
            toast.error(verifyResponse.data.message);
          }
        } catch (error) {
          console.error(error);
          toast.error(error.message);
        }
      },
    };
    const rzp = new window.Razorpay(options);
    rzp.open();
  };

  const onSubmitHandle = async (event) => {
    event.preventDefault();
    try {
      let orderItems = [];
      for (const itemId in cartItems) {
        for (const size in cartItems[itemId]) {
          if (cartItems[itemId][size] > 0) {
            const itemInfo = { ...products.find((p) => p._id === itemId) };
            if (itemInfo) {
              itemInfo.size = size;
              itemInfo.quantity = cartItems[itemId][size];
              orderItems.push(itemInfo);
            }
          }
        }
      }

      const orderData = {
        address: formData,
        items: orderItems,
        amount: getCartAmount() + delivery_fee,
      };

      switch (method) {
        case "cod":
          const response = await axios.post(
            backendUrl + "/api/order/place",
            orderData,
            { headers: { token } }
          );
          if (response.data.success) {
            setCartItems({});
            navigate("/orders");
          } else {
            toast.error(response.data.message);
          }
          break;

        case "stripe":
          const orderDataWithOrigin = {
            ...orderData,
            origin: window.location.origin,
          };
          const responseStripe = await axios.post(
            backendUrl + "/api/order/stripe",
            orderDataWithOrigin,
            { headers: { token } }
          );
          if (responseStripe.data.success) {
            const { session_url } = responseStripe.data;
            window.location.replace(session_url);
          } else {
            toast.error(responseStripe.data.message);
          }
          break;

        case "razorpay":
          const responseRazorpay = await axios.post(
            backendUrl + "/api/order/razorpay",
            orderData,
            { headers: { token } }
          );
          if (responseRazorpay.data.success) {
            initPay(responseRazorpay.data.order);
          } else {
            toast.error(responseRazorpay.data.message);
          }
          break;

        case "vnpay":
          const responseVnpay = await axios.post(
            backendUrl + "/api/order/vnpay",
            orderData,
            { headers: { token } }
          );
          if (responseVnpay.data.success) {
            window.location.replace(responseVnpay.data.paymentUrl);
          } else {
            toast.error(responseVnpay.data.message);
          }
          break;

        case "momo":
          const responseMomo = await axios.post(
            backendUrl + "/api/order/momo",
            orderData,
            { headers: { token } }
          );
          if (responseMomo.data.success) {
            window.location.replace(responseMomo.data.paymentUrl);
          } else {
            toast.error(responseMomo.data.message);
          }
          break;

        case "solana":
          const responseSolana = await axios.post(
            backendUrl + "/api/order/solana",
            orderData,
            { headers: { token } }
          );
          if (responseSolana.data.success) {
            setSolanaPaymentData(responseSolana.data.paymentData);
            setCurrentOrderId(responseSolana.data.orderId);
            setShowSolanaPayment(true);
          } else {
            toast.error(responseSolana.data.message);
          }
          break;

        default:
          break;
      }
    } catch (error) {
      console.error(error);
      toast.error(error.message);
    }
  };

  const handleSolanaPaymentSuccess = () => {
    setCartItems({});
    navigate("/orders");
  };

  const handleSolanaPaymentCancel = () => {
    setShowSolanaPayment(false);
    setSolanaPaymentData(null);
    setCurrentOrderId(null);
  };

  if (showSolanaPayment && solanaPaymentData) {
    return (
      <div className="flex justify-center items-center min-h-[80vh] pt-14">
        <SolanaPayment
          paymentData={solanaPaymentData}
          orderId={currentOrderId}
          backendUrl={backendUrl}
          token={token}
          onPaymentSuccess={handleSolanaPaymentSuccess}
          onPaymentCancel={handleSolanaPaymentCancel}
        />
      </div>
    );
  }

  return (
    <form
      onSubmit={onSubmitHandle}
      className="flex flex-col sm:flex-row justify-between gap-4 pt-5 sm:pt-14 min-h-[80vh] border-t"
    >
      <div className="flex flex-col gap-4 w-full sm:max-w-[480px]">
        <Title text1="DELIVERY" text2="INFORMATION" />
        <div className="flex gap-3">
          <input
            onChange={onChangeHandle}
            name="firstName"
            value={formData.firstName}
            className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
            type="text"
            placeholder="First name"
          />
          <input
            onChange={onChangeHandle}
            name="lastName"
            value={formData.lastName}
            className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
            type="text"
            placeholder="Last name"
          />
        </div>
        <input
          onChange={onChangeHandle}
          name="email"
          value={formData.email}
          className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
          type="email"
          placeholder="Email address"
        />
        <input
          onChange={onChangeHandle}
          name="street"
          value={formData.street}
          className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
          type="text"
          placeholder="Street"
        />
        <div className="flex gap-3">
          <input
            onChange={onChangeHandle}
            name="city"
            value={formData.city}
            className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
            type="text"
            placeholder="City"
          />
          <input
            onChange={onChangeHandle}
            name="state"
            value={formData.state}
            className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
            type="text"
            placeholder="State"
          />
        </div>
        <div className="flex gap-3">
          <input
            onChange={onChangeHandle}
            name="zipcode"
            value={formData.zipcode}
            className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
            type="number"
            placeholder="Zipcode"
          />
          <input
            onChange={onChangeHandle}
            name="country"
            value={formData.country}
            className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
            type="text"
            placeholder="Country"
          />
        </div>
        <input
          onChange={onChangeHandle}
          name="phone"
          value={formData.phone}
          className="border border-gray-300 rounded py-1.5 px-3.5 w-full"
          type="number"
          placeholder="Phone"
        />
      </div>
      <div className="mt-8">
        <CartTotal />
        <Title text1="PAYMENT" text2="METHOD" />
        <div className="flex gap-3 flex-col lg:flex-row">
          <div
            onClick={() => setMethod("stripe")}
            className="flex items-center gap-3 border p-2 px-3 cursor-pointer"
          >
            <p
              className={`min-w-3 h-3.5 border rounded-full ${method === "stripe" ? "bg-green-400" : ""
                }`}
            ></p>
            <img className="h-5 mx-4" src={assets.stripe_logo} alt="" />
          </div>
          <div
            onClick={() => setMethod("vnpay")}
            className="flex items-center gap-3 border p-2 px-3 cursor-pointer"
          >
            <p
              className={`min-w-3 h-3.5 border rounded-full ${method === "vnpay" ? "bg-green-400" : ""
                }`}
            ></p>
            <img className="h-5 mx-4" src={assets.vnpay_logo} alt="" />
          </div>
          <div
            onClick={() => setMethod("solana")}
            className="flex items-center gap-3 border p-2 px-3 cursor-pointer"
          >
            <p
              className={`min-w-3 h-3.5 border rounded-full ${
                method === "solana" ? "bg-green-400" : ""
              }`}
            ></p>
            <p className="text-gray-500 text-sm font-medium mx-4">SOLANA</p>
          </div>
          <div
            onClick={() => setMethod("cod")}
            className="flex items-center gap-3 border p-2 px-3 cursor-pointer"
          >
            <p
              className={`min-w-3 h-3.5 border rounded-full ${method === "cod" ? "bg-green-400" : ""
                }`}
            ></p>
            <p className="text-gray-500 text-sm font-medium mx-4">
              CASH ON DELIVERY
            </p>
          </div>
        </div>
        <div className="w-full text-end mt-8">
          <button
            type="submit"
            className="bg-black text-white px-16 py-3 text-sm"
          >
            PLACE ORDER
          </button>
        </div>
      </div>
    </form>
  );
};

export default PlaceOrder;
