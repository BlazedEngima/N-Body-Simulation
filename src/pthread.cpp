#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"

int n_thd; // number of threads

int n_body;
int n_iteration;

double total_time;

typedef struct {
    double x, y;  // position
    double vx, vy;  // velocity
    double m;  // mass
    double ax, ay; // acceleration
} my_Body;

void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n, my_Body *pool) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % (max_mass - min_mass) + min_mass;
        // x[i] = rand() % bound_x;
        // y[i] = rand() % bound_y;
        
        // For collision logic (fast paced)
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);

        // x[i] = 3000.0f + rand() % (bound_x / 4);
        // y[i] = 3000.0f + rand() % (bound_y / 4);

        vx[i] = 0.0f;
        vy[i] = 0.0f;

        pool[i].m = m[i];
        pool[i].x = x[i];
        pool[i].y = y[i];
        pool[i].vx = vx[i];
        pool[i].vy = vy[i];
        pool[i].ax = pool[i].ay = 0.0;
    }
}

void wall_collision(my_Body& body){
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

typedef struct {
    // Arguments for threads
    int start_body;
    int body_num;
    my_Body* ori_body;
    my_Body* new_body;

} Args;


void* worker(void* args) {
    // Main thread routine
    Args* my_arg = (Args*) args;
    int s_body = my_arg->start_body;
    int num_body = my_arg->body_num;
    int e_body = s_body + num_body;

    my_Body* ori_body = my_arg->ori_body;
    my_Body* new_body = my_arg->new_body;

    for (int i = s_body; i < e_body; i++) {
        ori_body[i].ax = ori_body[i].ay = 0.0;
        new_body[i] = ori_body[i]; 

        for (int j = 0; j < n_body; j++) {
            if (i == j) continue;
            interaction(ori_body[i],new_body[i],ori_body[j]);
        }
        update_data(new_body[i]);
    }
}


void master(){
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    my_Body* ori_body = new my_Body[n_body];

    generate_data(m, x, y, vx, vy, n_body, ori_body);

    Logger l = Logger("Pthread", n_body, bound_x, bound_y);

    for (int i = 0; i < n_iteration; i++) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        //Assigning Jobs
        pthread_t thds[n_thd]; // thread pool
        Args args[n_thd]; // arguments for all threads
        my_Body* new_body = new my_Body[n_body]; // store the body data after the iteration
        int temp_sum = 0;
        for (int thd = 0; thd < n_thd; thd++) {
            args[thd].start_body = temp_sum;
            if (thd == n_thd - 1) {
                args[thd].body_num = n_body - temp_sum;
            }
            else {
                args[thd].body_num = n_body / n_thd;
            }
            temp_sum += args[thd].body_num;
            args[thd].ori_body = ori_body;
            args[thd].new_body = new_body;
        }
        
        for (int thd = 0; thd < n_thd; thd++) pthread_create(&thds[thd], NULL, worker, &args[thd]);

        for (int thd = 0; thd < n_thd; thd++) pthread_join(thds[thd], NULL);
        
        for (int i = 0; i < n_body; i++){
            ori_body[i] = new_body[i];
            m[i] = ori_body[i].m;
            x[i] = ori_body[i].x;
            y[i] = ori_body[i].y;
            vx[i] = ori_body[i].vx;
            vy[i] = ori_body[i].vy;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = t2 - t1;

        printf("Iteration %d, elapsed time: %.3f\n", i, time_span);
        total_time += time_span.count();

        l.save_frame(x, y);

        #ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        double xi;
        double yi;
        for (int i = 0; i < n_body; i++){
            xi = x[i];
            yi = y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();
        #else

        #endif
    }

    delete[] ori_body;
    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
}


int main(int argc, char *argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_thd = atoi(argv[3]);

    #ifdef GUI
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Pthread");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    total_time = 0.0;
    master();

    printf("Student ID: 119010545\n"); // replace it with your student id
    printf("Name: Samuel Theofie\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation Pthread Implementation\n");
    printf("Total running time: %.3f\n",total_time);
	return 0;
}

