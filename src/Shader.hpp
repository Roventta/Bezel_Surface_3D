#ifndef RASTERIZER_SHADER_H
#define RASTERIZER_SHADER_H

#include <eigen3/Eigen/Eigen>

struct fragment_shader_payload
{
    fragment_shader_payload() = default;

    fragment_shader_payload(const Eigen::Vector3f& col, const Eigen::Vector3f& nor) :
         color(col), normal(nor) {}

    Eigen::Vector3f view_pos;
    Eigen::Vector3f color;
    Eigen::Vector3f normal;
    Eigen::Vector3f barycentric_coord;

};

struct vertex_shader_payload
{
    Eigen::Vector3f position;
};

#endif //RASTERIZER_SHADER_H
