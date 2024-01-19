#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f> &normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f *_v)
{
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f f0, f1, f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(float(x) + 0.5, float(y) + 0.5, 1.);
    if ((p.dot(f0) * f0.dot(v[2]) > 0) && (p.dot(f1) * f1.dot(v[0]) > 0) && (p.dot(f2) * f2.dot(v[1]) > 0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return {c1, c2, c3};
}


void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList)
{

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto &t : TriangleList)
    {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{
            (view * model * t->v[0]),
            (view * model * t->v[1]),
            (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v)
                       { return v.template head<3>(); });

        Eigen::Vector4f v[] = {
            mvp * t->v[0],
            mvp * t->v[1],
            mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v)
        {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
            inv_trans * to_vec4(t->normal[0], 0.0f),
            inv_trans * to_vec4(t->normal[1], 0.0f),
            inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        // newtri.setColor(0, 148, 121.0, 92.0);
        // newtri.setColor(1, 148, 121.0, 92.0);
        // newtri.setColor(2, 148, 121.0, 92.0);

        newtri.setColor(0, t->color[0][0] * 255, t->color[0][1] * 255, t->color[0][2] * 255);
        newtri.setColor(1, t->color[1][0] * 255, t->color[1][1] * 255, t->color[1][2] * 255);
        newtri.setColor(2, t->color[2][0] * 255, t->color[2][1] * 255, t->color[2][2] * 255);


        // Also pass view space vertice position
        rasterize_triangle(newtri, viewspace_pos);
    }
}


static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1, const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1, const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}


// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &view_pos)
{
    //    * v[i].w() is the vertex view space depth value z.
    //    * Z is interpolated view space depth for the current pixel
    //    * zp is depth between zNear and zFar, used for z-buffer
    auto v = t.toVector4();

    Vector4i bbox;
    bbox[0] = std::min(v[0][0], std::min(v[1][0], v[2][0]));
    bbox[1] = std::min(v[0][1], std::min(v[1][1], v[2][1]));
    bbox[2] = std::max(v[0][0], std::max(v[1][0], v[2][0]));
    bbox[3] = std::max(v[0][1], std::max(v[1][1], v[2][1]));

    // iterate through the pixel and find if the current pixel is inside the triangle
    for (int y = bbox[1]; y < bbox[3] + 1; y++)
    {
        for (int x = bbox[0]; x < bbox[2] + 1; x++)
        {
            if (insideTriangle(x, y, t.v))
            {
                float alpha, beta, gamma = 0;
                std::tie(alpha,beta,gamma) = computeBarycentric2D(x+0.5, y+0.5, t.v);
                float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                zp *= Z;
                int idx = get_index(x, y);
                if (zp < depth_buf[idx])
                {
                    depth_buf[idx] = zp;

                    // TODO: Interpolate the attributes, you can copy code from your HW3:
                    // auto interpolated_color
                    // auto interpolated_normal
                    // auto interpolated_shadingcoords

                    Vector3f interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);
                    Vector3f interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                    Vector3f interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                    //end of inter

                    fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized());
                    payload.view_pos = interpolated_shadingcoords;
                    payload.barycentric_coord = Eigen::Vector3f(alpha, beta, gamma);
                    // Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
                    auto pixel_color = fragment_shader(payload);
                    set_pixel(Vector2i(x, y), pixel_color);
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m)
{
    model = m;
}

void rst::rasterizer::mult_rot_trans(float angle_in_rad, Vector3f axis){
    Eigen::MatrixXf rotation(3,3);
    rotation = AngleAxisf(angle_in_rad, axis).toRotationMatrix();
    rotation.conservativeResize(4,4);
    rotation(3,3) = 1;
    model = rotation * model;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - y) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    if (point.x() < 0 || point.x() >= width || point.y() < 0 || point.y() >= height)
    {
        return;
    }

    // old index: auto ind = point.y() + point.x() * width;
    int ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}
