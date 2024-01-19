#include <opencv2/opencv.hpp>
#include "global.hpp"
#include "rasterizer.hpp"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle_in_rad, Vector3f axis)
{
    Eigen::MatrixXf rotation(3,3);
    rotation = AngleAxisf(angle_in_rad, axis).toRotationMatrix();
    rotation.conservativeResize(4,4);
    rotation(3,3) = 1;
    /*angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;
    */

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};


static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

Eigen::Vector3f wireframe_fragment_shader(const fragment_shader_payload& payload)
{
    const float edge_threshold = 0.02f; // distance at which we color the fragment as an edge

    Eigen::Vector3f gray(0.5, 0.5, 0.5);  // triangle interior
    Eigen::Vector3f white(1.0, 1.0, 1.0); // edges

    // Barycentric coordinates of the fragment
    float alpha = payload.barycentric_coord.x();
    float beta = payload.barycentric_coord.y();
    float gamma = payload.barycentric_coord.z();

    // fragment is close to any of the triangle's edges
    if (alpha < edge_threshold || beta < edge_threshold || gamma < edge_threshold)
    {
        return white * 255.f;
    }

    // inside the triangle
    return gray * 255.f;
}


Eigen::Vector3f xAxis_fragment_shader(const fragment_shader_payload& payload){
    Vector3f out = {255,0,0};
    return out;
}

Eigen::Vector3f yAxis_fragment_shader(const fragment_shader_payload& payload){
    Vector3f out = {0,255,0};
    return out;
}

Eigen::Vector3f zAxis_fragment_shader(const fragment_shader_payload& payload){
    Vector3f out = {0,0,255};
    return out;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    //std::cout<<kd<<"\n"<<std::endl;

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        float light_distance_squared=0;
        for(int i = 0; i<3; i++){
            float tmp = (point[i]-light.position[i]);
            light_distance_squared += tmp*tmp;
        }
        //std::cout<<point<<"\n" << std::endl;
        Vector3f lightV = (light.position-point)/sqrt(light_distance_squared);
        //Vector3f lightV = (light.position-point).normalized();
        float dot_prod = normal.dot(lightV);
        if(dot_prod<0){dot_prod = 0;}

        Vector3f bisector = ((eye_pos-point).normalized() + lightV).normalized();
        float speculat_dot_prod = normal.dot(bisector);
        if(speculat_dot_prod<0){speculat_dot_prod=0;}
        speculat_dot_prod =  pow(speculat_dot_prod,p);
        //diffuse
        Vector3f Ld = {0,0,0};
        //ambient
        Vector3f La = {0,0,0};
        //specular
        Vector3f Ls = {0,0,0};
        for(int i=0; i<3; i++){
            float pow_arrived = light.intensity[i]/light_distance_squared;
            Ld[i] = kd[i]*(pow_arrived)*dot_prod;
            La[i] = ka[i]*amb_light_intensity[i];
            Ls[i] = ks[i]*pow_arrived*speculat_dot_prod;
        }

        
        result_color+=Ld+La+Ls;

    }

    return result_color * 255.f;
}



Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f persp_to_ortho = Eigen::Matrix4f::Identity();
    persp_to_ortho << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -(zNear * zFar),
        0, 0, 1, 0;

    float eye_fov_rad = eye_fov * MY_PI / 180.0;

    float top = -zNear * tan(eye_fov_rad);
    float bottom = -top;
    float right = aspect_ratio * top;
    float left = -right;

    Eigen::Matrix4f ortho_translate = Eigen::Matrix4f::Identity();
    ortho_translate << 1, 0, 0, -(right + left) / 2,
        0, 1, 0, -(top + bottom) / 2,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;

    Eigen::Matrix4f ortho_scale = Eigen::Matrix4f::Identity();
    ortho_scale << 2 / (right - left), 0, 0, 0,
        0, 2 / (top - bottom), 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;

    projection = ortho_scale * ortho_translate * persp_to_ortho * projection;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}



std::vector<Triangle*> createHardSurface(const std::vector<std::vector<Eigen::Vector3f>>& control_grid) {

    std::vector<Triangle*> TriangleList;

    std::vector<Eigen::Vector3f> control_points;
    for (const auto& row : control_grid) {
        for (const auto& point : row) {
            control_points.push_back(point);
        }
    }

    if (control_points.size() != 16) {
        std::cerr << "Error: Expected 16 control points but received " << control_points.size() << std::endl;
        return TriangleList;
    }

    // Iterate over the 4x4 grid and create triangles
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
        
            Eigen::Vector3f p1 = control_points[i * 4 + j];
            Eigen::Vector3f p2 = control_points[i * 4 + j + 1];
            Eigen::Vector3f p3 = control_points[(i + 1) * 4 + j];
            Eigen::Vector3f p4 = control_points[(i + 1) * 4 + j + 1];

            // Create two triangles for each grid cell
            Triangle* tri1 = new Triangle();
            tri1->setVertex(0, Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0));
            tri1->setVertex(1, Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0));
            tri1->setVertex(2, Eigen::Vector4f(p3.x(), p3.y(), p3.z(), 1.0));

            Triangle* tri2 = new Triangle();
            tri2->setVertex(0, Eigen::Vector4f(p3.x(), p3.y(), p3.z(), 1.0));
            tri2->setVertex(1, Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0));
            tri2->setVertex(2, Eigen::Vector4f(p4.x(), p4.y(), p4.z(), 1.0));

            // Calculate normals
            Eigen::Vector3f normal1 = ((p2 - p1).cross(p3 - p1)).normalized();
            Eigen::Vector3f normal2 = ((p2 - p3).cross(p4 - p3)).normalized();

            tri1->setNormal(0, normal1);
            tri1->setNormal(1, normal1);
            tri1->setNormal(2, normal1);

            tri2->setNormal(0, normal2);
            tri2->setNormal(1, normal2);
            tri2->setNormal(2, normal2);

            // Set color
            tri1->setColor(0, 255, 0, 0);
            tri1->setColor(1, 255, 0, 0);
            tri1->setColor(2, 255, 0, 0);

            tri2->setColor(0, 255, 0, 0);
            tri2->setColor(1, 255, 0, 0);
            tri2->setColor(2, 255, 0, 0);

            // Add triangles to the list
            TriangleList.push_back(tri1);
            TriangleList.push_back(tri2);
        }
    }

    return TriangleList;
}


Eigen::Vector3f computeNormal(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& v3) {
    Eigen::Vector3f edge1 = v2 - v1;
    Eigen::Vector3f edge2 = v3 - v1;
    Eigen::Vector3f normal = edge1.cross(edge2);
    return normal.normalized();
}

Eigen::Vector4f toVector4f(const Eigen::Vector3f& vec3) {
    return Eigen::Vector4f(vec3[0], vec3[1], vec3[2], 1.0f);
}

//only works on uniform mesh
void fillTwoTriangle(Triangle* tri_ptr_l, Triangle* tri_ptr_r, int u, int v, std::vector<std::vector<Eigen::Vector3f>> surface_points){
    tri_ptr_l->setVertex(0, toVector4f(surface_points[v][u]));
    tri_ptr_l->setVertex(1, toVector4f(surface_points[v][u+1]));
    tri_ptr_l->setVertex(2, toVector4f(surface_points[v+1][u]));
    Vector3f tln = -computeNormal(surface_points[v][u], surface_points[v][u+1], surface_points[v+1][u]);
    tri_ptr_l->setNormal(0, tln);
    tri_ptr_l->setNormal(1, tln);
    tri_ptr_l->setNormal(2, tln);
    tri_ptr_l->setColor(0, 225,187,20);
    tri_ptr_l->setColor(1, 225,187,20);
    tri_ptr_l->setColor(2, 225,187,20);

    tri_ptr_r->setVertex(0, toVector4f(surface_points[v][u+1]));
    tri_ptr_r->setVertex(1, toVector4f(surface_points[v+1][u+1]));
    tri_ptr_r->setVertex(2, toVector4f(surface_points[v+1][u]));
    Vector3f trn = -computeNormal(surface_points[v][u+1], surface_points[v+1][u+1], surface_points[v+1][u]);
    tri_ptr_r->setNormal(0, trn);
    tri_ptr_r->setNormal(1, trn);
    tri_ptr_r->setNormal(2, trn);
    tri_ptr_r->setColor(0, 225,187,20);
    tri_ptr_r->setColor(1, 225,187,20);
    tri_ptr_r->setColor(2, 225,187,20);

    //printf("%d, %d\n", u, v);
    //std::cout<<surface_points[v][u]<<surface_points[v][u+1]<<surface_points[v+1][u]<<std::endl;
}


Eigen::Vector3f evaluate1DBezier(const std::vector<Eigen::Vector3f>& control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm for a single dimension
    // return Eigen::Vector3f();

    int stageNumber = control_points.size()-1;
    int points_per_stage = 0;

    int point_bucket_size = 0, bucket_idx=0;

    for(int i=1; i<= stageNumber; i++){
        point_bucket_size += i;
    }

    //printf("bucket size %d\n", point_bucket_size);

    Vector3f point_bucket[point_bucket_size];

    for(int p_idx=0; p_idx < stageNumber; p_idx++){
        //linear interpolation
        point_bucket[bucket_idx] = control_points[p_idx]*(1-t) + control_points[p_idx+1]*t;
        bucket_idx++;
    }

    int bucket_idx_offset = bucket_idx;
    int previous_idx_offset = 0;

    for(int s_idx=1; s_idx<stageNumber; s_idx++){
        points_per_stage = stageNumber-s_idx;
        //printf("stage %d\n", s_idx);

        for(int p_idx=0; p_idx < points_per_stage; p_idx++){
            /*printf("read %d and %d in list to creat %d\n", bucket_idx-bucket_idx_offset + previous_idx_offset 
                                                        , bucket_idx-bucket_idx_offset +previous_idx_offset +1
                                                        , bucket_idx);
            */
            point_bucket[bucket_idx] = (1-t) * point_bucket[bucket_idx-bucket_idx_offset + previous_idx_offset] 
                                        + point_bucket[bucket_idx-bucket_idx_offset+previous_idx_offset+1] * t;
            bucket_idx++;
        }

        previous_idx_offset = bucket_idx_offset;
        bucket_idx_offset += points_per_stage;
    }

    return point_bucket[point_bucket_size-1];

}



std::vector<Triangle*> generateBezierPatch(const std::vector<std::vector<Eigen::Vector3f>>& control_grid, int reso) 
{
    assert(control_grid.size() == 4 && control_grid[0].size() == 4);  // Ensure it's a 4x4 grid

    const int resolution = reso; // 10x10 grid
    std::vector<Triangle*> TriangleList;

    std::vector<std::vector<Eigen::Vector3f>> surface_points(resolution, std::vector<Eigen::Vector3f>(resolution));

    // TODO: Compute all the surface points using 1D de Casteljau's algorithm for both u and v dimensions using evaluate1DBezier()

    // assume mesh and control grid have identical width and height repectfully
    int sub_grid_dim = (resolution-1) / (control_grid.size()-1);

    //generate 4 control lines
    std::vector<std::vector<Eigen::Vector3f>> control_lines{control_grid.size(), std::vector<Eigen::Vector3f>(resolution)}; 
    for(int i=0; i<control_grid.size(); i++){
        //int u = i*sub_grid_dim;
        std::vector<Vector3f> cp{control_grid.size()};
        for(int cp_idx=0; cp_idx<control_grid.size(); cp_idx++){
            cp[cp_idx] = control_grid[cp_idx][i];
        }

        for(int v=0; v<resolution; v++){
            control_lines[i][v] = evaluate1DBezier(cp, float(v)/(resolution-1));
        }
    }

    
    for (int v=0; v<resolution; v++){
        std::vector<Vector3f> cp{control_grid.size()};
        for(int cp_idx=0; cp_idx<control_grid.size(); cp_idx++){
            cp[cp_idx] = control_lines[cp_idx][v];
        }

        for(int u=0; u<resolution; u++){
            surface_points[v][u] = evaluate1DBezier(cp, float(u)/(resolution-1));
        }

    }
    

    // TODO: Tessellate the surface_points into triangles and compute the normals, you can use computeNormal() to get the normal of the triangle

    // Examples of how to create a triangle and set the normal
    // Triangle* tri1 = new Triangle();
    // tri1->setVertex(0, Eigen::Vector4f(0, 0, 0, 1));
    // tri1->setVertex(1, Eigen::Vector4f(1, 0, 0, 1));
    // tri1->setVertex(2, Eigen::Vector4f(0, 1, 0, 1));
    // tri1->setNormal(0, Eigen::Vector3f(0, 0, 1));
    // tri1->setNormal(1, Eigen::Vector3f(0, 0, 1));
    // tri1->setNormal(2, Eigen::Vector3f(0, 0, 1));
    // TriangleList.push_back(tri1);
    
    for(int v=0; v<resolution-1; v++){
        for(int u=0; u<resolution-1; u++){
            Triangle* tri_l = new Triangle();
            Triangle* tri_r = new Triangle();

            fillTwoTriangle(tri_l, tri_r, u, v, surface_points);

            TriangleList.push_back(tri_l);
            TriangleList.push_back(tri_r);

        }
    }

    return TriangleList;
}

std::vector<Triangle*> generateAxisPatch(){
    std::vector<Triangle*> TriangleList;

    float w = 0.05;
    float l = 1.05;

    Triangle *x_l = new Triangle();
    x_l -> setVertex(0, Vector4f(l,w,0,1));
    x_l -> setVertex(1, Vector4f(-l,w,0,1));
    x_l -> setVertex(2, Vector4f(-l,-w,0,1));

    Triangle *x_r = new Triangle();
    x_r -> setVertex(0, Vector4f(l,w,0,1));
    x_r -> setVertex(1, Vector4f(-l,-w,0,1));
    x_r -> setVertex(2, Vector4f(l,-w,0,1));

    TriangleList.push_back(x_l);
    TriangleList.push_back(x_r);

    return TriangleList;
    
}

int main(int argc, const char** argv) 
{
    std::vector<Triangle*> TriangleList;
    std::string filename = "output.png";
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = wireframe_fragment_shader;

    if (argc == 2 && std::string(argv[1]) == "normal") 
    {
        active_shader = normal_fragment_shader;
    }
    else if (argc == 2 && std::string(argv[1]) == "phong") 
    {
        active_shader = phong_fragment_shader;
    }
    else if (argc == 2 && std::string(argv[1]) == "wireframe") 
    {
        active_shader = wireframe_fragment_shader;
    }
    else if (argc == 2 && std::string(argv[1]) == "1d_B_test"){
        std::vector<Vector3f> cp{{0,1,1}, {0,3,2}, {0,4,4}};
        std::cout << evaluate1DBezier(cp, 1) << std::endl;

        return 1; 
    } 

    rst::rasterizer r(700, 700);

    // Set shaders
    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);


    // Control points
    std::vector<std::vector<Eigen::Vector3f>> control_points = {
        {Eigen::Vector3f(-1.5, -1.0, -1.5), Eigen::Vector3f(-0.5, -0.6, -1.5), Eigen::Vector3f(0.5, -0.5, -1.5), Eigen::Vector3f(1.5, -0.9, -1.5)},
        {Eigen::Vector3f(-1.5, -0.7, -0.5), Eigen::Vector3f(-0.5, 0.8, -0.5),  Eigen::Vector3f(0.5, 0.7, -0.5),  Eigen::Vector3f(1.5, -0.4, -0.5)},
        {Eigen::Vector3f(-1.5, -0.5, 0.5),  Eigen::Vector3f(-0.5, 0.6, 0.5),   Eigen::Vector3f(0.5, 0.9, 0.5),   Eigen::Vector3f(1.5, -0.6, 0.5)},
        {Eigen::Vector3f(-1.5, -0.8, 1.5), Eigen::Vector3f(-0.5, -0.9, 1.5),  Eigen::Vector3f(0.5, -0.8, 1.5),  Eigen::Vector3f(1.5, -1.1, 1.5)}
    };

    
    //TriangleList = createHardSurface(control_points);

    // Comment out the above line and uncomment the below line to test your implementation of Bezier Surface 
    int reso = 10;
    TriangleList = generateBezierPatch(control_points, reso); 

    std::vector<Triangle*> axisTriangleList = generateAxisPatch();

    Eigen::Vector3f eye_pos = {0, 0, 10};

    int key = 0;
    float angleY = 0;

    float rotUnit = 0.025;

    Vector3f cp_var_unit = {0,0.025,0};
    Vector3f camera_var_unit = {0,0,0.025};

    int cp_index = 0;
    int cp_index_u = 0;
    int cp_index_v = 0;

    while (key != 27)
    {
        //if(key != 255){
        //printf("key stroke: %d\n", key);
        //}

        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angleY, Vector3f::UnitY())); // You might want to adjust this if needed
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
        r.set_fragment_shader(active_shader);
        r.draw(TriangleList);

        //draw axises
        r.set_fragment_shader(xAxis_fragment_shader);
        //r.mult_rot_trans(angleY, Vector3f::UnitY());
        r.draw(axisTriangleList);

        r.set_fragment_shader(yAxis_fragment_shader);
        r.set_model(get_model_matrix(0.5*M_PI, Vector3f::UnitZ()));
        r.mult_rot_trans(angleY, Vector3f::UnitY());
        r.draw(axisTriangleList);

        r.set_fragment_shader(zAxis_fragment_shader);
        r.set_model(get_model_matrix(0.5*M_PI, Vector3f::UnitY()));
        r.mult_rot_trans(angleY, Vector3f::UnitY());
        r.draw(axisTriangleList);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        
        key = cv::waitKey(10);


        //move forward or away from the mesh
        if(key == 82){
            eye_pos -= camera_var_unit;
            //std::cout << eye_pos << "\n" << std::endl;
        }

        if(key == 84){
            eye_pos += camera_var_unit;
            //std::cout << eye_pos << "\n" << std::endl;
        }

        // Rotate the mesh
        if (key == 81)
        {
            angleY -= rotUnit*M_PI;
        }
        else if (key == 83)
        {
            angleY += rotUnit*M_PI;
        }

        if (key == 115){
            cv::imwrite(filename, image); // save the image when 's' is pressed
        }

        if(key==117){
            control_points[cp_index_u][cp_index_v] += cp_var_unit;
            TriangleList = generateBezierPatch(control_points, reso); 
        }
        if(key==108){
            control_points[cp_index_u][cp_index_v] -= cp_var_unit;
            TriangleList = generateBezierPatch(control_points, reso); 
        }

        if(key==45){//+ for go to next control point
            cp_index--;
        }
        if(key==61){//- for precious point
            cp_index++;
        }
        
        if(key==46){
            reso += 3;
            printf("%d\n", reso);
            TriangleList = generateBezierPatch(control_points, reso);
        }
        if(key==44){
            reso -= 3;
            if(reso < 4){reso = 4;}
            printf("%d\n", reso);
            TriangleList = generateBezierPatch(control_points, reso);
        }

        if(key==32){
            angleY = 0;
            eye_pos = {0,0,10};
        }

        cp_index = cp_index % (control_points[0].size() * control_points.size());
        cp_index_u = cp_index / control_points[0].size();
        cp_index_v = cp_index % control_points[0].size();

        if(key==45 || key==61){printf("u:%d, v:%d\n", cp_index_u, cp_index_v);}

        // TODO: Add more interaction here
    }

    // Clean up
    for (Triangle* t : TriangleList)
    {
        delete t;
    }

    return 0;

}
