#include "cvgl.h"

using namespace std;
using namespace cv;

Matx<float, 4, 4> ModelView;
Matx<float, 4, 4> Viewport;
Matx<float, 4, 4> Projection;

void viewport(int x, int y, int w, int h)
{
	Viewport = Matx<float, 4, 4>::eye();
	Viewport(0, 3) = x + w / 2.f;
	Viewport(1, 3) = y + h / 2.f;
	Viewport(2, 3) = 255.f / 2.f;

	Viewport(0, 0) = w / 2.f;
	Viewport(1, 1) = h / 2.f;
	Viewport(2, 2) = 255.f / 2.f;
}

void projection(float coeff)
{
	Projection = Matx<float, 4, 4>::eye();
	Projection(3, 2) = coeff;
}

void lookat(Vec3f eye, Vec3f center, Vec3f up)
{
	Vec3f z = normalize(eye - center);
	Vec3f x = normalize(up.cross(z));
	Vec3f y = normalize(z.cross(x));
	Matx44f Minv = Matx44f::eye();
	Matx44f Tr = Matx44f::eye();
	for (int i = 0; i < 3; i++) {
		Minv(0, i) = x[i];
		Minv(1, i) = y[i];
		Minv(2, i) = z[i];
		Tr(i, 3) = -center[i];
	}
	ModelView = Minv * Tr;
}

Vec3f barycentric(Vec4f A, Vec4f B, Vec4f C, Vec2i P)
{
	Vec3f s[2];
	for (int i = 1; i >= 0; i--) {
		s[i][0] = C[i] - A[i];
		s[i][1] = B[i] - A[i];
		s[i][2] = A[i] - P[i];
	}
	Vec3f u = s[0].cross(s[1]);
	Vec3f ans = Vec3f(1, -1, 1);
	if (abs(u[2]) > 1e-2)
		ans = Vec3f(1.f - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]);
	return ans;
}

void triangle(Vec4f *pts, cvglShader* shader, Mat &image, Mat &zbuffer)
{
	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			bboxmin[j] = std::min(bboxmin[j], pts[i][j] / pts[i][3]);
			bboxmax[j] = std::max(bboxmax[j], pts[i][j] / pts[i][3]);
		}
	}
	Vec2i P = Vec2i(0, 0);
	for (P[0] = bboxmin[0]; P[0] <= bboxmax[0]; P[0]++) {
		for (P[1] = bboxmin[1]; P[1] <= bboxmax[1]; P[1]++) {
			if (P[1] < 0 || P[1] >= image.rows || P[0] < 0 || P[0] >= image.cols) continue;

			Vec3f c = barycentric(pts[0] / pts[0][3], pts[1] / pts[1][3], pts[2] / pts[2][3], P);
			if (c[0] < 0 || c[1] < 0 || c[2] < 0) continue;
			float z = Vec3f(pts[0][2], pts[1][2], pts[2][2]).dot(c);
			float w = Vec3f(pts[0][3], pts[1][3], pts[2][3]).dot(c);
			float frag_depth =z / w / 255;

			Scalar color;
			if (zbuffer.at<float>(P[1], P[0]) < frag_depth && !shader->fragment(c, color))
			{
				zbuffer.at<float>(P[1], P[0]) = frag_depth;
				image.at<Vec4b>(P[1], P[0]) = color;
			}

		}
	}
}


