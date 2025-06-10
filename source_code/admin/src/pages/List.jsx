import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import axios from "axios";
import { backendUrl, currency } from "../App";
import { toast } from "react-toastify";

const List = ({ token }) => {
  const [list, setList] = useState([]);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingProduct, setEditingProduct] = useState(null);
  const [editForm, setEditForm] = useState({
    name: "",
    category: "",
    price: "",
    description: "",
    type: "",
    sizes: [],
    bestseller: false,
  });

  const categories = ["Men (Nam)", "Women (Nữ)", "Kids (Trẻ em)"];
  const types = [
    "Topwear (Thun)",
    "Bottomwear (Quần dài)",
    "Winterwear (Áo khoác)",
  ];

  const fetchList = async () => {
    try {
      const response = await axios.get(backendUrl + "/api/product/list");
      if (response.data.products) {
        setList(response.data.products);
      } else {
        toast.error(response.data.products);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  };

  const handleEdit = (product) => {
    setEditingProduct(product);
    setEditForm({
      _id: product._id,
      name: product.name,
      category: product.category,
      price: product.price,
      description: product.description || "",
      type: product.type || "",
      sizes: product.sizes || [],
      bestseller: product.bestseller || false,
    });
    setShowEditModal(true);
  };

  const handleUpdate = async (e) => {
    e.preventDefault();
    if (!editingProduct?._id) {
      toast.error("Invalid product ID");
      return;
    }

    try {
      const updateData = {
        _id: editingProduct._id,
        name: editForm.name,
        category: editForm.category,
        price: Number(editForm.price),
        description: editForm.description,
        type: editForm.type,
        sizes: JSON.stringify(editForm.sizes),
        bestseller: Boolean(editForm.bestseller),
        image: editingProduct.image,
      };

      const response = await axios.post(
        backendUrl + "/api/product/update",
        updateData,
        {
          headers: {
            "Content-Type": "application/json",
            token: token,
          },
        }
      );

      if (response.data.success) {
        toast.success("Product updated successfully");
        await fetchList();
        setShowEditModal(false);
      } else {
        toast.error(response.data.message || "Failed to update product");
      }
    } catch (error) {
      console.error("Update error:", error);
      if (error.response) {
        toast.error(error.response.data.message || "Error updating product");
      } else if (error.request) {
        toast.error("No response from server");
      } else {
        toast.error("Error: " + error.message);
      }
    }
  };

  const removeProduct = async (id) => {
    try {
      const response = await axios.post(
        backendUrl + "/api/product/remove",
        { id },
        { header: { token } }
      );

      if (response.data.success) {
        toast.success(response.data.message);
        await fetchList();
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  };

  useEffect(() => {
    fetchList();
  }, []);

  return (
    <div>
      <p className="mb-2">Danh sách sản phẩm</p>
      <div className="flex flex-col gap-2">
        {/* List Table Title */}
        <div className="hidden md:grid grid-cols-[1fr_3fr_1fr_1fr_1fr_1fr] items-center py-1 px-2 border bg-gray-100 text-sm">
          <b>Ảnh</b>
          <b>Tên sản phẩm</b>
          <b>Danh mục</b>
          <b>Giá bán</b>
          <b className="text-center">Sửa</b>
          <b className="text-center">Xóa sản phẩm</b>
        </div>

        {/* Product List */}
        {list.map((item, index) => (
          <div
            className="grid grid-cols-[1fr_3fr_1fr] md:grid-cols-[1fr_3fr_1fr_1fr_1fr_1fr] items-center gap-2 py-1 px-2 border text-sm"
            key={index}
          >
            <img
              className="w-12 h-12 object-cover rounded"
              src={item.image[0]}
              alt=""
              onError={(e) => {
                e.target.onerror = null;
                e.target.src =
                  "https://via.placeholder.com/100x100?text=No+Image";
              }}
            />
            <p>{item.name}</p>
            <p>{item.category}</p>
            <p>
              {item.price} {currency}
            </p>
            <p
              onClick={() => handleEdit(item)}
              className="text-center cursor-pointer text-blue-500"
            >
              ✏️
            </p>
            <p
              onClick={() => removeProduct(item._id)}
              className="text-center cursor-pointer text-red-500"
            >
              🗑️
            </p>
          </div>
        ))}
      </div>

      {/* Edit Modal */}
      {showEditModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center overflow-y-auto">
          <div className="bg-white p-6 rounded-lg w-[600px] my-8 max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">Sửa thông tin sản phẩm</h2>
            <form onSubmit={handleUpdate}>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">
                  Tên sản phẩm
                </label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={(e) =>
                    setEditForm({ ...editForm, name: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                  required
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">
                  Danh mục
                </label>
                <select
                  value={editForm.category}
                  onChange={(e) =>
                    setEditForm({ ...editForm, category: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                  required
                >
                  {categories.map((category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  ))}
                </select>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Type</label>
                <select
                  value={editForm.type}
                  onChange={(e) =>
                    setEditForm({ ...editForm, type: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                >
                  {types.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">
                  Giá bán
                </label>
                <input
                  type="number"
                  value={editForm.price}
                  onChange={(e) =>
                    setEditForm({ ...editForm, price: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                  required
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Mô tả</label>
                <textarea
                  value={editForm.description}
                  onChange={(e) =>
                    setEditForm({ ...editForm, description: e.target.value })
                  }
                  className="w-full p-2 border rounded"
                  rows="3"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Sizes</label>
                <div className="flex gap-3">
                  <div
                    onClick={() =>
                      setEditForm((prev) => ({
                        ...prev,
                        sizes: prev.sizes.includes("S")
                          ? prev.sizes.filter((item) => item !== "S")
                          : [...prev.sizes, "S"],
                      }))
                    }
                  >
                    <p
                      className={`${
                        editForm.sizes.includes("S")
                          ? "bg-green-500"
                          : "bg-slate-200"
                      } px-3 py-1 cursor-pointer`}
                    >
                      S
                    </p>
                  </div>

                  <div
                    onClick={() =>
                      setEditForm((prev) => ({
                        ...prev,
                        sizes: prev.sizes.includes("M")
                          ? prev.sizes.filter((item) => item !== "M")
                          : [...prev.sizes, "M"],
                      }))
                    }
                  >
                    <p
                      className={`${
                        editForm.sizes.includes("M")
                          ? "bg-green-500"
                          : "bg-slate-200"
                      } px-3 py-1 cursor-pointer`}
                    >
                      M
                    </p>
                  </div>

                  <div
                    onClick={() =>
                      setEditForm((prev) => ({
                        ...prev,
                        sizes: prev.sizes.includes("L")
                          ? prev.sizes.filter((item) => item !== "L")
                          : [...prev.sizes, "L"],
                      }))
                    }
                  >
                    <p
                      className={`${
                        editForm.sizes.includes("L")
                          ? "bg-green-500"
                          : "bg-slate-200"
                      } px-3 py-1 cursor-pointer`}
                    >
                      L
                    </p>
                  </div>

                  <div
                    onClick={() =>
                      setEditForm((prev) => ({
                        ...prev,
                        sizes: prev.sizes.includes("XL")
                          ? prev.sizes.filter((item) => item !== "XL")
                          : [...prev.sizes, "XL"],
                      }))
                    }
                  >
                    <p
                      className={`${
                        editForm.sizes.includes("XL")
                          ? "bg-green-500"
                          : "bg-slate-200"
                      } px-3 py-1 cursor-pointer`}
                    >
                      XL
                    </p>
                  </div>

                  <div
                    onClick={() =>
                      setEditForm((prev) => ({
                        ...prev,
                        sizes: prev.sizes.includes("XXL")
                          ? prev.sizes.filter((item) => item !== "XXL")
                          : [...prev.sizes, "XXL"],
                      }))
                    }
                  >
                    <p
                      className={`${
                        editForm.sizes.includes("XXL")
                          ? "bg-green-500"
                          : "bg-slate-200"
                      } px-3 py-1 cursor-pointer`}
                    >
                      XXL
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={editForm.bestseller}
                    onChange={(e) =>
                      setEditForm({ ...editForm, bestseller: e.target.checked })
                    }
                    className="mr-2"
                  />
                  <span className="text-sm font-medium">
                    Thêm vào hàng bán chạy
                  </span>
                </label>
              </div>

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowEditModal(false)}
                  className="px-4 py-2 border rounded hover:bg-gray-100"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Cập nhật
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default List;
