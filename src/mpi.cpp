#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"


int n_body;
int n_iteration;


int my_rank;
int world_size;

double total_time;

typedef struct {
    double x, y;  // position
    double vx, vy;  // velocity
    double m;  // mass
    double ax, ay; // acceleration
} my_Body;

MPI_Datatype MPI_BODY;

void create_mpi_type() {
	MPI_Datatype array_of_types[7] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
	int blocklen[7] = {1,1,1,1,1,1,1};
	MPI_Aint array_of_displacements[7];

	my_Body temp;
	MPI_Aint addr_start, addresses[7];
	MPI_Get_address(&temp.x,&addresses[0]);
	MPI_Get_address(&temp.y,&addresses[1]);
	MPI_Get_address(&temp.vx,&addresses[2]);
	MPI_Get_address(&temp.vy,&addresses[3]);
    MPI_Get_address(&temp.ax,&addresses[4]);
	MPI_Get_address(&temp.ay,&addresses[5]);
	MPI_Get_address(&temp.m,&addresses[6]);
	MPI_Get_address(&temp,&addr_start);

	// calculate byte displacement of each block (array of address integer)
	array_of_displacements[0] = addresses[0] - addr_start;
	array_of_displacements[1] = addresses[1] - addr_start;
	array_of_displacements[2] = addresses[2] - addr_start;
	array_of_displacements[3] = addresses[3] - addr_start;
	array_of_displacements[4] = addresses[4] - addr_start;
	array_of_displacements[5] = addresses[5] - addr_start;
	array_of_displacements[6] = addresses[6] - addr_start;

	MPI_Type_create_struct(7,blocklen,array_of_displacements,array_of_types,&MPI_BODY);
	MPI_Type_commit(&MPI_BODY);
}

void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n) {
    // Generate proper initial position and mass for better visualization
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % (max_mass - min_mass) + min_mass;
        // x[i] = rand() % bound_x;
        // y[i] = rand() % bound_y;
        
        // For collision logic (fast paced)
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);

        // vx[i] = 0.0f;
        // vy[i] = 0.0f;
    }
}

void wall_collision(my_Body& body) {
    double r = sqrt(radius2);
    if (body.x <= r) {
        body.x = r + err;
        body.vx = -body.vx;
    } else if (body.x >= bound_x - r) {
        body.x = bound_x - r - err;
        body.vx = -body.vx;
    }
    if (body.y <= r) {
        body.y = r + err;
        body.vy = -body.vy;
    } else if (body.y >= bound_y - r) {
        body.y = bound_y - r - err;
        body.vy = - body.vy;
    }
}

void update_data(my_Body& body) {
    body.vx += body.ax * dt;
    body.vy += body.ay * dt;
    wall_collision(body);
    body.x += body.vx * dt;
    body.y += body.vy * dt;
    wall_collision(body);
}

void interaction(my_Body& ori, my_Body& ori_new, my_Body& ori_pair) {
    double delta_x = ori.x - ori_pair.x;
    double delta_y = ori.y - ori_pair.y;
    double dist_s = delta_x*delta_x + delta_y*delta_y;
    bool isCollision = false;

    if (dist_s <= radius2*4) {
        dist_s = radius2*4;
        isCollision = true;
    } // collision happens

    double dist = sqrt(dist_s);

    if (isCollision) {
        double dot_prod = delta_x * (ori.vx - ori_pair.vx) + delta_y * (ori.vy - ori_pair.vy);
        double value = 2 / (ori.m + ori_pair.m) * dot_prod / dist_s;
        ori_new.vx -= value * delta_x * ori_pair.m;
        ori_new.vy -= value * delta_y * ori_pair.m;
        
        ori_new.x += delta_x / dist * sqrt(radius2) / 2.0;
        ori_new.y += delta_y / dist * sqrt(radius2) / 2.0;

    } else {
        double F = ori.m*ori_pair.m*gravity_const / dist_s;
        ori_new.ax -= F * delta_x / ori.m;
        ori_new.ay -= F * delta_y / ori.m;
    }
}

int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double* total_m = new double[n_body];
    double* total_x = new double[n_body];
    double* total_y = new double[n_body];
    double* total_vx = new double[n_body];
    double* total_vy = new double[n_body];

    int local_size = (n_body + world_size - 1) / world_size;
    int total_size = local_size * world_size;
    my_Body* ori_data = new my_Body[total_size];
    my_Body* new_data = new my_Body[total_size];

    double* calc_x = new double[total_size];
    double* calc_y = new double[total_size];
    double* calc_vx = new double[total_size];
    double* calc_vy = new double[total_size];

    
	if (my_rank == 0) {
		#ifdef GUI
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(500, 500); 
		glutInitWindowPosition(0, 0);
		glutCreateWindow("N Body Simulation MPI Implementation");
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0, bound_x, 0, bound_y);
		#endif

		total_time = 0.0;

		generate_data(total_m, total_x, total_y, total_vx, total_vy, n_body);

		Logger l = Logger("MPI", n_body, bound_x, bound_y);

		for (int i = 0; i < n_iteration; i++) {
		    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		    MPI_Bcast(total_m,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_x,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_y,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_vx,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_vy,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);

		    for (int j = 0; j < total_size; j++) {
				if (j >= n_body) {
					ori_data[j].m = 0.0;
					ori_data[j].x = 0.0;
					ori_data[j].y = 0.0;
					ori_data[j].vx = 0.0;
					ori_data[j].vy = 0.0;
				} // pixels that do not exist.

				ori_data[j].m = total_m[j];
				ori_data[j].x = total_x[j];
				ori_data[j].y = total_y[j];
				ori_data[j].vx = total_vx[j];
				ori_data[j].vy = total_vy[j];
		    }

		    int offset = local_size*my_rank;
		    for (int j = offset; j < std::min(offset + local_size, n_body); j++) {
			ori_data[j].ax = ori_data[j].ay = 0.0;
			new_data[j] = ori_data[j]; 
			for (int p = 0; p < n_body; p++) {
			    if (p == j) continue;
			    interaction(ori_data[j],new_data[j],ori_data[p]);
			}
			update_data(new_data[j]);
		    }

		    double* local_x = new double[local_size];
		    double* local_y = new double[local_size];
		    double* local_vx = new double[local_size];
		    double* local_vy = new double[local_size];

		    for (int j = offset; j < std::min(offset + local_size, n_body); j++) {
				local_x[j-offset] = new_data[j].x;
				local_y[j-offset] = new_data[j].y;
				local_vx[j-offset] = new_data[j].vx;
				local_vy[j-offset] = new_data[j].vy;
		    }

		    MPI_Gather(local_x,local_size,MPI_DOUBLE,calc_x,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Gather(local_y,local_size,MPI_DOUBLE,calc_y,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Gather(local_vx,local_size,MPI_DOUBLE,calc_vx,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Gather(local_vy,local_size,MPI_DOUBLE,calc_vy,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);

		    delete[] local_x;
		    delete[] local_y;
		    delete[] local_vx;
		    delete[] local_vy;

		    for (int j = 0; j < n_body; j++) {
				total_x[j] = calc_x[j];
				total_y[j] = calc_y[j];
				total_vx[j] = calc_vx[j];
				total_vy[j] = calc_vy[j];
		    }

		    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		    std::chrono::duration<double> time_span = t2 - t1;

		    printf("Iteration %d, elapsed time: %.3f\n", i, time_span);
		    total_time += time_span.count();

		    l.save_frame(total_x, total_y);

		    #ifdef GUI
		    glClear(GL_COLOR_BUFFER_BIT);
		    glColor3f(1.0f, 0.0f, 0.0f);
		    glPointSize(2.0f);
		    glBegin(GL_POINTS);
		    double xi;
		    double yi;
		    for (int i = 0; i < n_body; i++) {
			xi = total_x[i];
			yi = total_y[i];
			glVertex2f(xi, yi);
		    }
		    glEnd();
		    glFlush();
		    glutSwapBuffers();
		    #else

		    #endif
		}

	} else {
		// MPI Routine
		for (int it = 0; it < n_iteration; it++) {
		    MPI_Bcast(total_m,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_x,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_y,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_vx,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(total_vy,n_body,MPI_DOUBLE,0,MPI_COMM_WORLD);

		    for (int i = 0; i < total_size; i++) {
			if (i >= n_body) {
			    ori_data[i].m = 0.0;
			    ori_data[i].x = 0.0;
			    ori_data[i].y = 0.0;
			    ori_data[i].vx = 0.0;
			    ori_data[i].vy = 0.0;
			} // pixels that do not exist.
			ori_data[i].m = total_m[i];
			ori_data[i].x = total_x[i];
			ori_data[i].y = total_y[i];
			ori_data[i].vx = total_vx[i];
			ori_data[i].vy = total_vy[i];
		    }

		    int offset = local_size*my_rank;
		    for (int j = offset; j < std::min(offset + local_size, n_body); j++){
			ori_data[j].ax = ori_data[j].ay = 0.0;
			new_data[j] = ori_data[j]; 
			for (int p = 0; p < n_body; p++){
			    if (p == j) continue;
			    interaction(ori_data[j],new_data[j],ori_data[p]);
			}
			update_data(new_data[j]);
		    }

		    double* local_x = new double[local_size];
		    double* local_y = new double[local_size];
		    double* local_vx = new double[local_size];
		    double* local_vy = new double[local_size];
		    for (int j = offset; j < std::min(offset + local_size, n_body); j++) {
			local_x[j-offset] = new_data[j].x;
			local_y[j-offset] = new_data[j].y;
			local_vx[j-offset] = new_data[j].vx;
			local_vy[j-offset] = new_data[j].vy;
		    }

		    MPI_Gather(local_x,local_size,MPI_DOUBLE,calc_x,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Gather(local_y,local_size,MPI_DOUBLE,calc_y,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Gather(local_vx,local_size,MPI_DOUBLE,calc_vx,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Gather(local_vy,local_size,MPI_DOUBLE,calc_vy,local_size,MPI_DOUBLE,0,MPI_COMM_WORLD);

		    delete[] local_x;
		    delete[] local_y;
		    delete[] local_vx;
		    delete[] local_vy;
		}
 	}

    delete[] total_m;
    delete[] total_x;
    delete[] total_y;
    delete[] total_vx;
    delete[] total_vy;

    delete[] calc_x;
    delete[] calc_y;
    delete[] calc_vx;
    delete[] calc_vy;

    delete[] ori_data;
    delete[] new_data;

	if (my_rank == 0) {
		printf("Student ID: 119010545\n"); // replace it with your student id
		printf("Name: Samuel Theofie\n"); // replace it with your name
		printf("Assignment 2: N Body Simulation MPI Implementation\n");
        	printf("Total running time: %.3f\n",total_time);
	}

	MPI_Finalize();

	return 0;
}

