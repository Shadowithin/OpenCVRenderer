#include<iostream>
#include<vector>
#include "model.h"
#include "cvgl.h"

#pragma warning(disable:4996)

using namespace std;
using namespace cv;

const int width = 800;
const int height = 800;

Model        *model = nullptr;
cvglShader  *shader = nullptr;
Mat           shadowbuffer;

Vec3f   light_dir(1, 1, 1);
Vec3f         eye(1.2, -.8, 3);
Vec3f      center(0, 0, 0);
Vec3f          up(0, 1, 0);

class ShadowShader : public cvglShader {
private:
	Vec2f  varying_uv[3];
	Vec4f  varying_tri[3]; 
	Vec3f  varying_nrm[3];
	Vec3f  ndc_tri[3];
public:
	Matx44f uniform_T;
	Matx44f uniform_TINV;
	Matx44f uniform_TSD;

	ShadowShader(Matx44f TSD)
	{
		uniform_T = Projection * ModelView;
		uniform_TINV = uniform_T.inv();
		uniform_TINV = uniform_TINV.t();
		uniform_TSD = TSD * ((Viewport * Projection * ModelView).inv());

		Vec4f light = Vec4f(0, 0, 0, 0);
		for (int i = 0; i < 3; i++) light[i] = light_dir[i];
		light = uniform_T * light;
		for (int i = 0; i < 3; i++) light_dir[i] = light[i];
	}

	ShadowShader()
	{
		uniform_T = Projection * ModelView;
		uniform_TINV = uniform_T.inv();
		uniform_TINV = uniform_TINV.t();

		Vec4f light = Vec4f(0, 0, 0, 0);
		for (int i = 0; i < 3; i++) light[i] = light_dir[i];
		light = uniform_T * light;
		for (int i = 0; i < 3; i++) light_dir[i] = light[i];
	}
	virtual ~ShadowShader() {}
	virtual Vec4f vertex(int iface, int nthvert) {
		varying_uv[nthvert] = model->uv(iface, nthvert);

		Vec3f n = model->normal(iface, nthvert);
		Vec4f norm = Vec4f(0, 0, 0, 0);
		for (int i = 0; i < 3; i++) norm[i] = n[i];
		norm = uniform_TINV * norm;
		for (int i = 0; i < 3; i++) varying_nrm[nthvert][i] = norm[i];

		Vec3f v = model->vert(iface, nthvert);
		Vec4f cvgl_Vertex = Vec4f(1, 1, 1, 1);
		for (int i = 0; i < 3; i++) cvgl_Vertex[i] = v[i];
		cvgl_Vertex =  Projection * ModelView * cvgl_Vertex;

		for (int i = 0; i < 3; i++)
			ndc_tri[nthvert][i] = cvgl_Vertex[i] / cvgl_Vertex[3];
		cvgl_Vertex = Viewport * cvgl_Vertex;
		varying_tri[nthvert] = cvgl_Vertex;

		return cvgl_Vertex;
	}

	virtual bool fragment(Vec3f bar, Scalar &color) {
		Vec4f sb_p = uniform_TSD * (varying_tri[0] * bar[0] + varying_tri[1] * bar[1] + varying_tri[2] * bar[2]);
		sb_p = sb_p / sb_p[3];
		float shadow = .3 + .7*(shadowbuffer.at<uchar>(int(sb_p[1]), int(sb_p[0])) < sb_p[2] + 1.34);

		Vec3f bn = normalize(varying_nrm[0] * bar[0] + varying_nrm[1] * bar[1] + varying_nrm[2] * bar[2]);
		Vec2f uv = varying_uv[0] * bar[0] + varying_uv[1] * bar[1]+ varying_uv[2] * bar[2];

		Matx<float, 3, 3> A;
		Vec3f a0 = ndc_tri[1] - ndc_tri[0];
		Vec3f a1 = ndc_tri[2] - ndc_tri[0];
		for (int i = 0; i < 3; i++)
		{
			A(0, i) = a0[i];
			A(1, i) = a1[i];
			A(2, i) = bn[i];
		}

		Matx<float, 3, 3> AI = A.inv();

		Matx<float, 3, 3> B;
		Vec3f b0 = normalize(AI * Vec3f(varying_uv[1][0] - varying_uv[0][0], varying_uv[2][0] - varying_uv[0][0], 0));
		Vec3f b1 = normalize(AI * Vec3f(varying_uv[1][1] - varying_uv[0][1], varying_uv[2][1] - varying_uv[0][1], 0));
	

		for (int i = 0; i < 3; i++)
		{
			B(i, 0) = b0[i];
			B(i, 1) = b1[i];
			B(i, 2) = bn[i];
		}

		Vec3f n = normalize(B * model->normal(uv));
		Vec3f r = normalize(n * (n.dot(light_dir) * 2.f) - light_dir);
		float spec = pow(max(r[2], 0.0f), model->specular(uv));
		float diff = std::max(0.f, n.dot(light_dir));

		Scalar c = model->diffuse(uv);
		for (int i = 0; i < 3; i++) color[i] = std::min<float>(0.1 + c[i] * shadow*(1.2*diff + .1*spec), 255);
		//color = model->diffuse(uv)*diff;

		return false;
	}
};

class DepthShader : public cvglShader {
private:
	Vec4f  varying_tri[3];
public:
	virtual Vec4f vertex(int iface, int nthvert) {
		Vec3f v = model->vert(iface, nthvert);
		Vec4f cvgl_Vertex = Vec4f(1, 1, 1, 1);
		for (int i = 0; i < 3; i++) cvgl_Vertex[i] = v[i];
		cvgl_Vertex = Viewport * Projection * ModelView * cvgl_Vertex;
		varying_tri[nthvert] = cvgl_Vertex;
		return cvgl_Vertex;
	}
	virtual bool fragment(Vec3f bar, Scalar &color) {
		Vec4f p = varying_tri[0] * bar[0] + varying_tri[1] * bar[1] + varying_tri[2] * bar[2];
		color = Scalar(255, 255, 255)*(p[2]/255.f);
		return false;
	}
};

int main()
{
	Mat frame = Mat::zeros(height, width, CV_8UC4);
	Mat depth = Mat::zeros(height, width, CV_8UC4);
	shadowbuffer = Mat::zeros(height, width, CV_8UC1);
	Mat zbuffer = Mat::zeros(height, width, CV_8UC1);
	model = new Model("./obj/diablo3_pose/diablo3_pose.obj");
	
	clock_t start = clock();

	{//depth
		light_dir = normalize(light_dir);
		lookat(light_dir, center, up);
		viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
		//viewport(0, 0, width, height);
		projection(-1.f / norm(eye - center));

		shader = new DepthShader();
		for (int i = 0; i < model->nfaces(); i++) {
			Vec4f screen_coords[3];
			for (int j = 0; j < 3; j++)
				screen_coords[j] = shader->vertex(i, j);
			triangle(screen_coords, shader, depth, shadowbuffer);
		}
		//imwrite("shadowbuffer.bmp", shadowbuffer);
		delete shader;
		shader = nullptr;
	}
	
	Matx44f MSD = Viewport * Projection * ModelView;

	lookat(eye, center, up);

	shader = new ShadowShader(MSD);
	for (int i = 0; i < model->nfaces(); i++) {
		Vec4f screen_coords[3];
		for (int j = 0; j < 3; j++)
			screen_coords[j] = shader->vertex(i, j);
		triangle(screen_coords, shader, frame, zbuffer);
	}

	clock_t end = clock();
	cout << end - start << endl;

	Mat framebuffer;
	flip(frame, framebuffer, 0);
	imshow("frame", framebuffer);
	imwrite("frame_diablo.jpg", framebuffer);
	delete model;
	delete shader;
	model = nullptr;
	shader = nullptr;
	waitKey(0);
	return 0;
}