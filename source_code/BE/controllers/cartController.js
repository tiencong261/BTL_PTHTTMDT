// import userModel from "../models/userModel.js";

// // add products to user cart
// const addToCart = async (req, res) => {
//     try{

//         const { userId, itemId, size} = req.body;

//         const userData = await userModel.findById(userId);
//         let cartData = await userData.cartData;

//         if(cartData[itemId]){
//             if(cartData[itemId][size]){
//                 cartData[itemId][size] += 1;
//             }
//             else{
//                 cartData[itemId][size] = 1;
//             }
//         }
//         else{
//             cartData[itemId] = {};
//             cartData[itemId][size] = 1;
//         }

//         await userModel.findByIdAndUpdate(userId, {cartData});

//         res.json({success:true, message:"Added To Cart"})

//     } catch (error){
//         console.log(error);
//         res.json({success:false, message:error.message});
//     }
// }

// // update user cart
// const updateCart = async (req, res) => {
//     try{
//         const { userId, itemId, size, quantity } = req.body;

//         const userData = await userModel.findById(userId);
//         let cartData = await userData.cartData;

//         cartData[itemId][size] = quantity;

//         await userModel.findByIdAndUpdate(userId, {cartData});

//         res.json({success:true, message:"Cart Updated"})
//     } catch(error){
//         console.log(error);
//         res.json({success:false, message:error.message});
//     }
// }

// // get user cart data
// const getUserCart = async (req, res) => {
//     try{

//         const {userId} = req.body;

//         const userData = await userModel.findById(userId);
//         let cartData = await userData.cartData;

//         res.json({success:true, cartData: userData.cartData});

//     } catch (error) {
//         console.log(error);
//         res.json({success:false, message:error.message});
//     }
// }

// export { addToCart, updateCart, getUserCart };

import userModel from "../models/userModel.js";
import productModel from "../models/productModel.js";

// add products to user cart
const addToCart = async (req, res) => {
  try {
    const { userId, itemId, size } = req.body;

    const userData = await userModel.findById(userId);
    let cartData = userData.cartData || {};

    if (cartData[itemId]) {
      if (cartData[itemId][size]) {
        cartData[itemId][size] += 1;
      } else {
        cartData[itemId][size] = 1;
      }
    } else {
      cartData[itemId] = {};
      cartData[itemId][size] = 1;
    }

    await userModel.findByIdAndUpdate(userId, { cartData });

    res.json({ success: true, message: "Added To Cart" });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

// update user cart
const updateCart = async (req, res) => {
  try {
    const { userId, itemId, size, quantity } = req.body;

    const userData = await userModel.findById(userId);
    let cartData = userData.cartData;

    if (cartData[itemId]) {
      cartData[itemId][size] = quantity;

      // Nếu quantity = 0 thì xóa size đó
      if (quantity === 0) {
        delete cartData[itemId][size];
        if (Object.keys(cartData[itemId]).length === 0) {
          delete cartData[itemId];
        }
      }

      await userModel.findByIdAndUpdate(userId, { cartData });
    }

    res.json({ success: true, message: "Cart Updated" });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

// get user cart data
const getUserCart = async (req, res) => {
  try {
    const { userId } = req.body;
    const userData = await userModel.findById(userId);
    let cartData = userData.cartData || {};

    // Lấy danh sách tất cả productId đang có trong cart
    const cartProductIds = Object.keys(cartData);

    // Tìm các productId còn tồn tại trong DB
    const validProducts = await productModel.find({
      _id: { $in: cartProductIds },
    });

    const validProductIds = validProducts.map((product) =>
      product._id.toString()
    );

    // Xóa các productId không còn tồn tại khỏi cartData
    let removedItems = [];
    for (let productId of cartProductIds) {
      if (!validProductIds.includes(productId)) {
        delete cartData[productId];
        removedItems.push(productId);
      }
    }

    // Nếu có sản phẩm bị xóa, cập nhật lại cart
    if (removedItems.length > 0) {
      await userModel.findByIdAndUpdate(userId, { cartData });
    }

    res.json({
      success: true,
      cartData,
      removedItems,
      message:
        removedItems.length > 0
          ? "Một số sản phẩm đã bị shop xóa khỏi hệ thống. Vui lòng chọn sản phẩm khác."
          : "Lấy giỏ hàng thành công.",
    });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

export { addToCart, updateCart, getUserCart };
