import React from "react";
import { assets } from "../assets/assets";
import Title from "../components/Title";
import NewsletterBox from "../components/NewsletterBox";

const About = () => {
  return (
    <div>
      <div className="text-2xl text-center pt-8 border-t">
        <Title text1={"GIỚI THIỆU"} text2={"VỀ CHÚNG TÔI"} />
      </div>

      <div className="my-10 flex flex-col md:flex-row gap-16">
        <img
          className="w-full md:max-w-[480px]"
          src={assets.about_img}
          alt=""
        />
        <div className="flex flex-col justify-center gap-6 md:w-2/4 text-gray-600">
          <p>
            Forever được ra đời từ niềm đam mê với sự đổi mới và mong muốn thay
            đổi cách mọi người mua sắm trực tuyến. Hành trình của chúng tôi bắt
            đầu với một ý tưởng đơn giản: xây dựng một nền tảng nơi khách hàng
            có thể dễ dàng khám phá, lựa chọn và mua sắm các sản phẩm thời trang
            ngay tại nhà một cách tiện lợi.
          </p>
          <p>
            Kể từ khi thành lập, chúng tôi không ngừng nỗ lực để tuyển chọn
            những mẫu trang phục chất lượng, đa dạng về phong cách nhằm đáp ứng
            mọi nhu cầu và gu thẩm mỹ của khách hàng. Từ thời trang thường ngày
            đến những bộ đồ cá tính, sang trọng – tất cả đều được lựa chọn kỹ
            lưỡng từ các thương hiệu uy tín.
          </p>
          <b className="text-gray-800">Mục tiêu của chúng tôi</b>
          <p>
            Forever hướng đến việc mang đến cho khách hàng sự lựa chọn phong
            phú, trải nghiệm mua sắm tiện lợi và cảm giác an tâm tuyệt đối.
            Chúng tôi cam kết xây dựng một hành trình mua sắm trọn vẹn – từ việc
            tìm kiếm sản phẩm, đặt hàng cho đến khâu giao hàng và dịch vụ sau
            bán hàng.
          </p>
        </div>
      </div>
      <div className="text-4xl py-4">
        <Title text1={"TẠI SAO"} text2={"CHỌN CHÚNG TÔI"} />
      </div>

      <div className="flex flex-col md:flex-row text-sm mb-20">
        <div className="border px-10 md:px-16 py-8 sm:py-20 flex flex-col gap-5">
          <b>Đảm Bảo Chất Lượng:</b>
          <p className="text-gray-600">
            Chúng tôi chọn lọc và kiểm tra kỹ lưỡng từng sản phẩm để đảm bảo đáp
            ứng các tiêu chuẩn chất lượng nghiêm ngặt của chúng tôi.
          </p>
        </div>
        <div className="border px-10 md:px-16 py-8 sm:py-20 flex flex-col gap-5">
          <b>Tiện Lợi:</b>
          <p>
            Với giao diện thân thiện và quy trình đặt hàng nhanh gọn, việc mua
            sắm chưa bao giờ dễ dàng đến thế.
          </p>
        </div>
        <div className="border px-10 md:px-16 py-8 sm:py-20 flex flex-col gap-5">
          <b>Dịch Vụ Khách Hàng Xuất Sắc:</b>
          <p>
            Đội ngũ chuyên nghiệp tận tâm của chúng tôi luôn sẵn sàng hỗ trợ
            bạn, đảm bảo sự hài lòng của bạn là ưu tiên hàng đầu.
          </p>
        </div>
      </div>

      <NewsletterBox />
    </div>
  );
};

export default About;
