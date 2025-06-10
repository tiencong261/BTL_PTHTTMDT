import { createContext, useEffect, useState } from "react";

// import { products } from "../assets/assets";
import { useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "react-toastify";
import axios from "axios";

export const ShopContext = createContext();

const ShopContextProvider = (props) => {
  const currency = "VND";
  const delivery_fee = 10000;
  const backendUrl = import.meta.env.VITE_BACKEND_URL;

  const [search, setSearch] = useState("");
  const [showSearch, setShowSearch] = useState(true);
  const [cartItems, setCartItems] = useState({});
  const [products, setProducts] = useState([]);
  const [token, setToken] = useState("");
  const [shownMissingProductWarning, setShownMissingProductWarning] =
    useState(false);
  const [user, setUser] = useState(null);

  const navigate = useNavigate();

  const addToCart = async (itemId, size) => {
    if (!size) {
      toast.error("Select Product Size");
      return;
    }

    let cartData = structuredClone(cartItems);
    if (cartData[itemId]) {
      cartData[itemId][size] = (cartData[itemId][size] || 0) + 1;
    } else {
      cartData[itemId] = { [size]: 1 };
    }

    setCartItems(cartData);

    if (token) {
      try {
        await axios.post(
          backendUrl + "/api/cart/add",
          { itemId, size },
          { headers: { token } }
        );
      } catch (error) {
        console.log(error);
        toast.error(error.message);
      }
    }
  };

  const getCartCount = () => {
    let total = 0;
    for (const itemId in cartItems) {
      const product = products.find((p) => p._id === itemId);
      if (!product) continue;

      for (const size in cartItems[itemId]) {
        if (cartItems[itemId][size] && cartItems[itemId][size] > 0) {
          total += cartItems[itemId][size];
        }
      }
    }
    return total;
  };

  const getProductsData = async () => {
    try {
      const response = await axios.get(backendUrl + "/api/product/list");
      if (response.data.success) {
        setProducts(response.data.products);
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  };

  const updateQuantity = async (itemId, size, quantity) => {
    let cartData = structuredClone(cartItems);

    if (quantity === 0) {
      if (cartData[itemId] && cartData[itemId][size] !== undefined) {
        delete cartData[itemId][size];

        if (Object.keys(cartData[itemId]).length === 0) {
          delete cartData[itemId];
        }
      }
    } else {
      if (!cartData[itemId]) {
        cartData[itemId] = {};
      }
      cartData[itemId][size] = quantity;
    }

    setCartItems(cartData);

    if (token) {
      try {
        await axios.post(
          backendUrl + "/api/cart/update",
          { itemId, size, quantity },
          { headers: { token } }
        );
      } catch (error) {
        console.log(error);
        toast.error(error.message);
      }
    }
  };

  const getCartAmount = () => {
    let totalAmount = 0;
    for (const itemId in cartItems) {
      const product = products.find((p) => p._id === itemId);
      if (product) {
        for (const size in cartItems[itemId]) {
          if (cartItems[itemId][size] > 0) {
            totalAmount += product.price * cartItems[itemId][size];
          }
        }
      }
    }
    return totalAmount;
  };

  const getUserCart = async (token) => {
    try {
      const response = await axios.post(
        backendUrl + "/api/cart/get",
        {},
        { headers: { token } }
      );
      if (response.data.success) {
        setCartItems(response.data.cartData);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  };

  const getUserProfile = async (token) => {
    try {
      const response = await axios.get(backendUrl + "/api/user/profile", {
        headers: { token },
      });
      if (response.data.success) {
        setUser(response.data.user);
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error("Không thể lấy thông tin người dùng");
    }
  };

  useEffect(() => {
    if (token) {
      getUserProfile(token);
    }
  }, [token]);

  // Lấy dữ liệu sản phẩm khi load lần đầu
  useEffect(() => {
    getProductsData();
  }, []);

  // Lấy giỏ hàng khi có token và sản phẩm đã được tải
  useEffect(() => {
    if (token && products.length > 0) {
      getUserCart(token);
    }
  }, [token, products]);

  // Khôi phục token từ localStorage
  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    if (!token && storedToken) {
      setToken(storedToken);
    }
  }, []);

  // Kiểm tra nếu có sản phẩm trong giỏ đã bị xóa thì hiển thị cảnh báo
  useEffect(() => {
    if (products.length === 0 || Object.keys(cartItems).length === 0) return;

    let updatedCart = { ...cartItems };
    let cartChanged = false;
    let hasMissingProduct = false;

    for (const itemId in cartItems) {
      const productExists = products.find((p) => p._id === itemId);
      if (!productExists) {
        delete updatedCart[itemId];
        cartChanged = true;
        hasMissingProduct = true;
      }
    }

    if (cartChanged) {
      setCartItems(updatedCart);
    }

    if (hasMissingProduct && !shownMissingProductWarning) {
      toast.warn(
        "Sản phẩm trong giỏ hàng của bạn hiện không còn tồn tại. Vui lòng chọn sản phẩm khác."
      );
      setShownMissingProductWarning(true);
    }
  }, [cartItems, products]);

  const value = {
    products,
    currency,
    delivery_fee,
    search,
    setSearch,
    showSearch,
    setShowSearch,
    cartItems,
    addToCart,
    setCartItems,
    getCartCount,
    updateQuantity,
    getCartAmount,
    navigate,
    backendUrl,
    setToken,
    token,
    user,
    setUser,
  };

  return (
    <ShopContext.Provider value={value}>{props.children}</ShopContext.Provider>
  );
};

export default ShopContextProvider;
