#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"

int n_body;
int n_iteration;

double total_time;

void generate_data(double *m, double *x,double *y,double *vx,double *vy, int n) {
    // TODO: Generate proper initial position and mass for better visualization
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % max_mass + 1.0f;
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);
        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}

void update_position(double *x, double *y, double *vx, double *vy, int n) {
    //TODO: update position 
    for (int i = 0; i < n; i++)
    {
        x[i] += vx[i]*dt;
        y[i] += vy[i]*dt;

        double r = sqrt(radius2);
        if (x[i] <= r){
            x[i] = r + err;
            vx[i] = -vx[i];
        }
        else if (x[i] >= bound_x-r){
            x[i] = bound_x - r - err;
            vx[i] = -vx[i];
        }
        if (y[i] <= r){
            y[i] = r + err;
            vy[i] = -vy[i];
        }
        else if (y[i] >= bound_y-r){
            y[i] = bound_y - r - err;
            vy[i] = -vy[i];
        } // handle wall collision
    }
}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n) {
    //TODO: calculate force and acceleration, update velocity
    double* Fx = new double[n];
    double* Fy = new double[n];
    for(int i = 0; i < n; i++){
        Fx[i] = 0.0;
        Fy[i] = 0.0;
    } // initialize the acceleration

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            double delta_x = x[i] - x[j];
            double delta_y = y[i] - y[j];
            double dist_s = delta_x*delta_x + delta_y*delta_y;
            bool isCollision = false;
            if (dist_s <= radius2*4){
                dist_s = radius2*4;
                isCollision = true;
            } // collision happens
            double dist = sqrt(dist_s);

            if (isCollision){
                double dot_product = delta_x*(vx[i]-vx[j]) + delta_y*(vy[i]-vy[j]);
                double value = 2 / (m[i]+m[j]) * dot_product / dist_s;
                vx[i] -= value * delta_x * m[j];
                vy[i] -= value * delta_y * m[j];
                vx[j] += value * delta_x * m[i];
                vy[j] += value * delta_y * m[i]; // conservation of momentum

                x[i] += delta_x / dist * sqrt(radius2) / 2.0;
                y[i] += delta_y / dist * sqrt(radius2) / 2.0;
                x[j] -= delta_x / dist * sqrt(radius2) / 2.0;
                y[j] -= delta_y / dist * sqrt(radius2) / 2.0; // considering the collision volume
            } // if collision happened, ignore the force between the collision pair.
            else{
                double F = m[i]*m[j]*gravity_const / dist_s; // calculate the force
                Fx[i] -= (F / m[i]) * delta_x;
                Fy[i] -= (F / m[i]) * delta_y;
                Fx[j] += (F / m[j]) * delta_x;
                Fy[j] += (F / m[j]) * delta_y;
            } // update the component of acceleration
        }
    } // calculate acceleration
    
    for (int i = 0; i < n; i++){
        vx[i] += Fx[i]*dt;
        vy[i] += Fy[i]*dt;
    } // update the velocity
    
    delete[] Fx;
    delete[] Fy;

    for (int i = 0; i < n; i++){
        double r = sqrt(radius2);
        if (x[i] <= r){
            x[i] = r + err;
            vx[i] = -vx[i];
        }
        else if (x[i] >= bound_x-r){
            x[i] = bound_x - r - err;
            vx[i] = -vx[i];
        }
        if (y[i] <= r){
            y[i] = r + err;
            vy[i] = -vy[i];
        }
        else if (y[i] >= bound_y-r){
            y[i] = bound_y - r - err;
            vy[i] = -vy[i];
        }
    } // handle wall collsion (consider the collison volumn between pixels)
}


void master() {
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    Logger l = Logger("sequential", n_body, bound_x, bound_y);

    for (int i = 0; i < n_iteration; i++){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        update_velocity(m, x, y, vx, vy, n_body);
        update_position(x, y, vx, vy, n_body);

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

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    
}


int main(int argc, char *argv[]){
    
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation Sequential Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    total_time = 0.0;
    master();

    printf("Student ID: 119010437\n"); // replace it with your student id
    printf("Name: ZHANG Shiyi\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation Sequential Implementation\n");
    printf("Total running time: %.3f\n",total_time);
    
    return 0;

}


