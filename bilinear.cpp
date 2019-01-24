#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <dirent.h>
#include <sys/types.h>

# define LG_PI              ( 3.14159265358979323846264338327950 )
# define LG_PI2 			( 6.28318530717958647692528676655901 )
# define LG_ATN(x,y)        ( ( ( x ) >= 0 ) ? ( ( ( y ) >= 0 ) ? atan( ( y ) / ( x ) ) : LG_PI2 + atan( ( y ) / ( x ) ) ) : LG_PI + atan( ( y ) / ( x ) ) )
# define LG_ASN(x)          ( asin( x ) )
# define LG_ACS(x)          ( acos( x ) )

using namespace cv;
using namespace std;

int main(){
	/* parameters */
	float azim;
	float elev;
	float roll;
	float angle_ouvert=45;
	/* square images generation for simplicity of matrix K */
	int pix_factor=2000;
	int width_output=pix_factor;
	int height_output=pix_factor;
	/* Calculated from "angle_ouvert" and "width_output=height_output=pix_factor" */
	float focal=0.5*pix_factor/tan(0.5*angle_ouvert*LG_PI/180.0);
	/* Read the folder "data" */
	struct dirent *entry;
	DIR *dir = opendir("data/");
	while((entry=readdir(dir))!=NULL){
		std::string fullname=entry->d_name;
		size_t lastindex=fullname.find_last_of("."); 
		std::string name=fullname.substr(0,lastindex); 
		std::stringstream ss1;
		ss1 << "data/" << name << ".jpg";
		std::string image_input_name=ss1.str();
		if(image_input_name!="data/.jpg" && image_input_name!="data/..jpg"){
			/* input image read */
			Mat image_input;
			image_input=imread(image_input_name,CV_LOAD_IMAGE_COLOR);
			if(!image_input.data){
				cout << "Could not open or find the image" << std::endl;
			};
			for(int D_azim=0;D_azim<8;D_azim++){
				azim=LG_PI*D_azim/4;
				elev=0.0;
				roll=0.0;
				Mat image_output(height_output,width_output,CV_8UC3);
				/* rotation matrix setting */
				float Rot_Mat[3][3]={
					{1.0,0.0,0.0},
					{0.0,1.0,0.0},
					{0.0,0.0,1.0}
				};
				float const CosA=cos(azim);
				float const SinA=sin(azim);
				float const CosE=cos(elev);
				float const SinE=sin(elev);
				float const CosR=cos(roll);
				float const SinR=sin(roll);
				Rot_Mat[0][0]=CosA*CosE;
				Rot_Mat[0][1]=CosA*SinE*SinR-SinA*CosR;
				Rot_Mat[0][2]=CosA*SinE*CosR+SinA*SinR;
				Rot_Mat[1][0]=SinA*CosE; 
				Rot_Mat[1][1]=SinA*SinE*SinR+CosA*CosR;
				Rot_Mat[1][2]=SinA*SinE*CosR-CosA*SinR;
				Rot_Mat[2][0]=-SinE;
				Rot_Mat[2][1]=CosE*SinR;
				Rot_Mat[2][2]=CosE*CosR;
				
				float Pvi[3]={0.0,0.0,0.0};
				float Pvf[3]={0.0,0.0,0.0};
				float SX;
				float SY;
				/* loop for the output image */
				for(int DY=0;DY<height_output;DY++){
					for(int DX=0;DX<width_output;DX++){
						/* Compute pixel position in 3d-frame */
						Pvi[0]=focal;
						Pvi[1]=DX-0.5*width_output;
						Pvi[2]=DY-0.5*height_output;

						/* Compute rotated pixel position in 3d-frame */
						Pvf[0]=Rot_Mat[0][0]*Pvi[0]+Rot_Mat[0][1]*Pvi[1]+Rot_Mat[0][2]*Pvi[2];
						Pvf[1]=Rot_Mat[1][0]*Pvi[0]+Rot_Mat[1][1]*Pvi[1]+Rot_Mat[1][2]*Pvi[2];
						Pvf[2]=Rot_Mat[2][0]*Pvi[0]+Rot_Mat[2][1]*Pvi[1]+Rot_Mat[2][2]*Pvi[2];

						/* Retrieve mapping pixel (x,y)-coordinates */
						float norme_Pvf;
						norme_Pvf=sqrt(Pvf[0]*Pvf[0]+Pvf[1]*Pvf[1]+Pvf[2]*Pvf[2]);
						SX=image_input.cols*(LG_ATN(Pvf[0],Pvf[1])/(2*LG_PI));
						SY=image_input.rows*(LG_ASN(Pvf[2]/norme_Pvf)/LG_PI+0.5);
						
						/* INTERPOLATION setting*/
						float X1;
						float X2;
						float Y1;
						float Y2;
						
						X1=floor(SX);
						X2=ceil(SX);
						Y1=floor(SY);
						Y2=ceil(SY);
						
						Vec3b F11=image_input.at<Vec3b>(Point(X1,Y1));
						Vec3b F12=image_input.at<Vec3b>(Point(X1,Y2));
						Vec3b F21=image_input.at<Vec3b>(Point(X2,Y1));
						Vec3b F22=image_input.at<Vec3b>(Point(X2,Y2));
						
						Vec3b color;
						float E1;
						float E2;
						float E3;
						float E4;
						
						/* INTERPOLATION formula */
						for(int i=0;i<3;i++){
							if(X2==X1 && Y2==Y1){
								color[i]=F11[i];
							}else if(Y2==Y1){
								color[i]=F11[i]+(F21[i]-F11[i])*((SX-X1)/(X2-X1));
							}else if(X2==X1){
								color[i]=F11[i]+(F12[i]-F11[i])*((SY-Y1)/(Y2-Y1));
							}else{
								E1=F11[i];
								E2=(F21[i]-F11[i])*((SX-X1)/(X2-X1));
								E3=(F12[i]-F11[i])*((SY-Y1)/(Y2-Y1));
								E4=(F11[i]+F22[i]-F21[i]-F12[i])*((SX-X1)/(X2-X1))*((SY-Y1)/(Y2-Y1));
								color[i]=E1+E2+E3+E4;
							}
						}
						/* color attribution */
						image_output.at<Vec3b>(Point(DX,DY))=color;
					}
				}
				/* output name generation */
				int azim_deg=(int)round(azim*180.0/LG_PI);
				int elev_deg=(int)round(elev*180.0/LG_PI);
				int roll_deg=(int)round(roll*180.0/LG_PI);
				std::string azim_str=std::to_string(azim_deg);
				std::string elev_str=std::to_string(elev_deg);
				std::string roll_str=std::to_string(roll_deg);
				std::stringstream ss2;
				ss2 << "data_recti/" << name << "_" << azim_str << ".jpg";
				std::string image_output_name=ss2.str();
				/* image writting and release memory */
				imwrite(image_output_name,image_output);
				image_output.release();
				cout << image_output_name << std::endl;
			}
			image_input.release();
		}
	}
	closedir(dir);
}
