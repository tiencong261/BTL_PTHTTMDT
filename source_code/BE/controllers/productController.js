import { v2 as cloudinary } from "cloudinary";
import productModel from "../models/productModel.js";

//function for add product
const addProduct = async (req, res) => {
  try {
    const {
      name,
      description,
      price,
      category,
      subCategory,
      sizes,
      bestseller,
    } = req.body;

    const image1 = req.files.image1 && req.files.image1[0];
    const image2 = req.files.image2 && req.files.image2[0];
    const image3 = req.files.image3 && req.files.image3[0];
    const image4 = req.files.image4 && req.files.image4[0];

    const images = [image1, image2, image3, image4].filter(
      (item) => item != undefined
    );

    let imagesUrl = await Promise.all(
      images.map(async (item) => {
        let result = await cloudinary.uploader.upload(item.path, {
          resource_type: "image",
        });
        return result.secure_url;
      })
    );

    const productData = {
      name,
      description,
      category,
      price: Number(price),
      subCategory,
      bestseller: bestseller === "true" ? true : "false",
      sizes: JSON.parse(sizes),
      image: imagesUrl,
      date: Date.now(),
    };

    console.log(productData);

    const product = new productModel(productData);
    await product.save();

    res.json({ success: true, message: "Product Added" });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

//listProducts
const listProducts = async (req, res) => {
  try {
    const products = await productModel.find({});
    res.json({ success: true, products });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

//removeProduct
const removeProduct = async (req, res) => {
  try {
    await productModel.findByIdAndDelete(req.body.id);
    res.json({ success: true, message: "Product Removed" });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

//single product
const singleProduct = async (req, res) => {
  try {
    const { productId } = req.body;
    const product = await productModel.findById(productId);
    res.json({ success: true, product });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

//updateProduct
const updateProduct = async (req, res) => {
  try {
    const {
      _id,
      name,
      description,
      price,
      category,
      subCategory,
      sizes,
      bestseller,
      image,
    } = req.body;

    const updated = await productModel.findByIdAndUpdate(_id, {
      name,
      description,
      price: Number(price),
      category,
      subCategory,
      sizes: JSON.parse(sizes),
      bestseller: bestseller === "true" ? true : false,
      image,
    });

    if (updated) {
      res.json({ success: true, message: "Cập nhật sản phẩm thành công" });
    } else {
      res.json({ success: false, message: "Không tìm thấy sản phẩm" });
    }
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

//import products from assets
const importProductsFromAssets = async (req, res) => {
  try {
    const { products } = req.body;

    // Validate products array
    if (!Array.isArray(products)) {
      return res.json({ success: false, message: "Products must be an array" });
    }

    // Insert all products
    const result = await productModel.insertMany(products);

    res.json({
      success: true,
      message: `Successfully imported ${result.length} products`,
      products: result,
    });
  } catch (error) {
    console.log(error);
    res.json({ success: false, message: error.message });
  }
};

export {
  addProduct,
  listProducts,
  removeProduct,
  singleProduct,
  updateProduct,
  importProductsFromAssets,
};
