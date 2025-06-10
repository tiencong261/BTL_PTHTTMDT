// import React, { useEffect, useState } from 'react'
// import { useSearchParams } from 'react-router-dom'
// import axios from 'axios'
// import { backendUrl } from '../App'
// import { toast } from 'react-toastify'

// const EditProduct = ({ token }) => {
//   const [searchParams] = useSearchParams();
//   const id = searchParams.get("id");

//   const [product, setProduct] = useState({
//     name: '',
//     category: '',
//     price: '',
//     image: ['']
//   });

//   const fetchProduct = async () => {
//     try {
//       const res = await axios.get(`${backendUrl}/api/product/${id}`);
//       if (res.data.product) {
//         setProduct(res.data.product);
//       } else {
//         toast.error("Không tìm thấy sản phẩm");
//       }
//     } catch (err) {
//       toast.error(err.message);
//     }
//   };

//   const updateProduct = async () => {
//     try {
//       const res = await axios.post(`${backendUrl}/api/product/update`, product, {
//         headers: { token }
//       });
//       if (res.data.success) {
//         toast.success("Cập nhật thành công!");
//         window.location.href = "/admin/products";
//       } else {
//         toast.error(res.data.message);
//       }
//     } catch (err) {
//       toast.error(err.message);
//     }
//   };

//   useEffect(() => {
//     fetchProduct();
//   }, []);

//   return (
//     <div className="max-w-md mx-auto p-4">
//       <h2 className="text-xl font-bold mb-4">Chỉnh sửa sản phẩm</h2>

//       <input value={product.name} onChange={(e) => setProduct({ ...product, name: e.target.value })}
//         placeholder="Tên sản phẩm" className="w-full mb-2 p-2 border rounded" />

//       <input value={product.category} onChange={(e) => setProduct({ ...product, category: e.target.value })}
//         placeholder="Loại sản phẩm" className="w-full mb-2 p-2 border rounded" />

//       <input value={product.price} onChange={(e) => setProduct({ ...product, price: e.target.value })}
//         placeholder="Giá" type="number" className="w-full mb-2 p-2 border rounded" />

//       <input value={product.image[0]} onChange={(e) => setProduct({ ...product, image: [e.target.value] })}
//         placeholder="Link ảnh" className="w-full mb-2 p-2 border rounded" />

//       <button onClick={updateProduct} className="bg-blue-600 text-white px-4 py-2 rounded mt-2">Cập nhật</button>
//     </div>
//   );
// };

// export default EditProduct;



import React, { useEffect, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { backendUrl } from '../App'
import { toast } from 'react-toastify'

const EditProduct = ({ token }) => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const id = searchParams.get("id");

  const [loading, setLoading] = useState(true);
  const [product, setProduct] = useState({
    name: '',
    category: '',
    price: '',
    image: ['']
  });

  const fetchProduct = async () => {
    try {
      const res = await axios.get(`${backendUrl}/api/product/${id}`);
      if (res.data.product) {
        setProduct(res.data.product);
        setLoading(false);
      } else {
        toast.error("Không tìm thấy sản phẩm");
        navigate('/admin/products');
      }
    } catch (err) {
      toast.error(err.message);
      navigate('/admin/products');
    }
  };

  const updateProduct = async () => {
    try {
      const res = await axios.post(`${backendUrl}/api/product/update`, product, {
        headers: { token }
      });
      if (res.data.success) {
        toast.success("Cập nhật thành công!");
        navigate("/admin/products");
      } else {
        toast.error(res.data.message);
      }
    } catch (err) {
      toast.error(err.message);
    }
  };


  return (
    <div className="max-w-md mx-auto p-4">
      <h2 className="text-xl font-bold mb-4">Chỉnh sửa sản phẩm</h2>

      {product.image[0] && (
        <img src={product.image[0]} alt="Preview" className="w-32 h-32 object-cover mb-2 border rounded" />
      )}

      <input
        value={product.name}
        onChange={(e) => setProduct({ ...product, name: e.target.value })}
        placeholder="Tên sản phẩm"
        className="w-full mb-2 p-2 border rounded"
      />

      <input
        value={product.category}
        onChange={(e) => setProduct({ ...product, category: e.target.value })}
        placeholder="Loại sản phẩm"
        className="w-full mb-2 p-2 border rounded"
      />

      <input
        value={product.price}
        onChange={(e) => setProduct({ ...product, price: e.target.value })}
        placeholder="Giá"
        type="number"
        className="w-full mb-2 p-2 border rounded"
      />

      <input
        value={product.image[0]}
        onChange={(e) => setProduct({ ...product, image: [e.target.value] })}
        placeholder="Link ảnh"
        className="w-full mb-2 p-2 border rounded"
      />

      <button
        onClick={updateProduct}
        className="bg-blue-600 text-white px-4 py-2 rounded mt-2 hover:bg-blue-700"
      >
        Cập nhật
      </button>
    </div>
  );
};

export default EditProduct;

