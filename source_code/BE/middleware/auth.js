import jwt from "jsonwebtoken";

const authUser = async (req, res, next) => {
  try {
    const { token } = req.headers;
    if (!token) {
      return res
        .status(401)
        .json({ success: false, message: "Authentication token is required" });
    }

    const token_decode = jwt.verify(token, process.env.JWT_SECRET);
    if (!token_decode || !token_decode.id) {
      return res.status(401).json({ success: false, message: "Invalid token" });
    }

    req.body.userId = token_decode.id;
    next();
  } catch (error) {
    if (error.name === "TokenExpiredError") {
      return res
        .status(401)
        .json({ success: false, message: "Token has expired" });
    }
    if (error.name === "JsonWebTokenError") {
      return res.status(401).json({ success: false, message: "Invalid token" });
    }
    console.error("Auth middleware error:", error);
    res.status(500).json({ success: false, message: "Internal server error" });
  }
};

export default authUser;
