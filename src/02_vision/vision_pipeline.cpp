#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

int main(int argc, char** argv) {
    cout << "[C++] Pokrećem PCL i Eigen obradu..." << endl;

    string input_path = "sirovi_oblak_baza.pcd";
    if (argc > 1) {
        input_path = argv[1];
    }

    cout << "[C++] Pokušavam učitati datoteku: " << input_path << endl;

    // =========================================================================
    // 1. KORAK: UČITAVANJE (SEGMENTIRANI OBLAK - REGISTRIRAN)
    // =========================================================================
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_path, *cloud) == -1) {
        PCL_ERROR("[C++] GREŠKA: PCL ne može pročitati datoteku!\n");
        return -1;
    }
    
    pcl::io::savePCDFileASCII("01_segmentirano_i_registrirano.pcd", *cloud);
    cout << "[C++] Učitan oblak sa " << cloud->points.size() << " tocaka." << endl;

    // =========================================================================
    // 2. KORAK: FILTRIRANJE (STATISTICAL OUTLIER REMOVAL)
    // =========================================================================
    int sor_mean_k = 20;
    double sor_std_thresh = 0.8;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(sor_mean_k);
    sor.setStddevMulThresh(sor_std_thresh);
    sor.filter(*cloud_filtered);
    
    string name_filtered = "02_filtrirano_SOR_k" + to_string(sor_mean_k) + "_std0.8.pcd";
    pcl::io::savePCDFileASCII(name_filtered, *cloud_filtered);
    cout << "[C++] Nakon filtriranja ostalo " << cloud_filtered->points.size() << " tocaka." << endl;

    // =========================================================================
    // 3. KORAK: GEOMETRIJSKO REZANJE
    // =========================================================================
    
    std::vector<double> z_values;
    for(const auto& pt : cloud_filtered->points) {
        z_values.push_back(pt.z);
    }
    
    std::sort(z_values.begin(), z_values.end());
    
    int rez_indeks = z_values.size() * 0.40;
    double z_prag = z_values[rez_indeks];
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZ>);
    for(const auto& pt : cloud_filtered->points) {
        if(pt.z > z_prag) {  
            cloud_cropped->points.push_back(pt);
        }
    }
    
    cloud_cropped->width = cloud_cropped->points.size();
    cloud_cropped->height = 1;
    cloud_cropped->is_dense = true; 
    
    string name_cropped = "03_konacno_rezano_Z-Prag40posto.pcd";
    pcl::io::savePCDFileASCII(name_cropped, *cloud_cropped);
    cout << "[C++] Spremljeno: " << name_cropped << " (" << cloud_cropped->points.size() << " tocaka)" << endl;

    // =========================================================================
    // 4. KORAK: LEAST SQUARES
    // =========================================================================
    int N = cloud_cropped->points.size();
    if(N < 10) {
        cerr << "[C++] GREŠKA: Nedovoljno točaka za fitanje sfere!" << endl;
        return -1;
    }

    Eigen::MatrixXd A(N, 4);
    Eigen::VectorXd b(N);

    for(int i = 0; i < N; ++i) {
        double x = cloud_cropped->points[i].x;
        double y = cloud_cropped->points[i].y;
        double z = cloud_cropped->points[i].z;
        A(i, 0) = 2.0 * x;
        A(i, 1) = 2.0 * y;
        A(i, 2) = 2.0 * z;
        A(i, 3) = 1.0;
        b(i) = x*x + y*y + z*z;
    }

    Eigen::VectorXd u = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    
    double cx = u(0);
    double cy = u(1);
    double cz = u(2);
    double w = u(3);
    
    double r = sqrt(w + cx*cx + cy*cy + cz*cz);

    cout << "[C++] Izračunat centar sfere: X=" << cx << ", Y=" << cy << ", Z=" << cz << endl;
    cout << "[C++] Izračunat radijus sfere: r=" << r << endl;

    ofstream out("tocan_centar_oraha.txt");
    out << cx << " " << cy << " " << cz << endl;
    out.close();

    // =========================================================================
    // VIZUALIZACIJA
    // =========================================================================
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Prikaz Fitane Sfere"));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud_cropped, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_cropped, cloud_color, "orah");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "orah");

    pcl::PointXYZ center_pt(cx, cy, cz);
    viewer->addSphere(center_pt, r, 1.0, 0.0, 0.0, "sfera"); // 1.0, 0.0, 0.0 je RGB crvena
    
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.4, "sfera");

    viewer->addCoordinateSystem(0.05);

    cout << "[C++] Otvaram 3D prozor! (Zatvori prozor na 'X' kako bi se program nastavio i robot krenuo...)" << endl;
    
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    cout << "[C++] Prozor zatvoren. Izlazim." << endl;
    return 0;
}