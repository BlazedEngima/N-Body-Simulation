#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <tuple>

#ifdef GUI
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "./headers/physics.h"
#include "./headers/logger.h"


int n_body;
int n_iteration;

int n_omp_threads;


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

        // x[i] = 3000.0f + rand() % (bound_x / 4);
        // y[i] = 3000.0f + rand() % (bound_y / 4);

        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}

void update_position(double *x, double *y, double *vx, double *vy, int i) {
    // Update position 
    
    x[i] = x[i] + (vx[i] * dt);
    y[i] = y[i] + (vy[i] * dt);

    double r = sqrt(radius2);

    // Border collision logic
    if (x[i] <= r) {
        x[i] = r + err;
        vx[i] = -vx[i];
    }
    else if (x[i] >= bound_x-r) {
        x[i] = bound_x - r - err;
        vx[i] = -vx[i];
    }
    if (y[i] <= r) {
        y[i] = r + err;
        vy[i] = -vy[i];
    }
    else if (y[i] >= bound_y-r) {
        y[i] = bound_y - r - err;
        vy[i] = -vy[i];
    }

    

}

void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int i) {
    double ax = 0;
    double ay = 0;
    
    // Calculate force and acceleration, update velocity 
    for (size_t j = 0; j < n_body; j++) {
        if (i == j) continue;

        // Calculate the acceleration
        double deltaX = x[i] - x[j];
        double deltaY = y[i] - y[j];
        double r_2 = (deltaX * deltaX) + (deltaY * deltaY);

        bool collision = false;

        // Detect collision
        if (r_2 <= radius2) {

            collision = true;
            r_2 = radius2;
        }

        double r = sqrt(r_2);

        // No collisions
        if (!collision) {
            double F = m[i]*m[j]*gravity_const / r_2; // calculate the force
            ax -= (F / m[i]) * deltaX;
            ay -= (F / m[i]) * deltaY;

        // Collision with other bodies ignore collision force
        } else {
            // Conservation of momentum
            double dot = deltaX * (vx[i] - vx[j]) + deltaY * (vy[i] - vy[j]);
            double val = 2 / (m[i] + m[j]) * dot / r_2;

            // Get velocity from the conservation of momentum
            vx[i] -= val * deltaX * m[j];
            vy[i] -= val * deltaY * m[j];

            // We consider the volume of a body and reduce that from the positions
            x[i] += deltaX / r * sqrt(radius2) / 2.0;
            y[i] += deltaY / r * sqrt(radius2) / 2.0;
        }
    }

    // Update velocity arrays
    vx[i] = vx[i] + (dt * ax);
    vy[i] = vy[i] + (dt * ay);
    
}

void master() {
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    Logger l = Logger("OpenMP", n_body, bound_x, bound_y);

    std::chrono::duration<double> total{};

    for (int i = 0; i < n_iteration; i++){
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        
        //Threads configuration
        omp_set_num_threads(n_omp_threads);
        #pragma omp parallel for
         for (int i = 0; i < n_body; i++) {
            update_velocity(m, x, y, vx, vy, i);
        }
        

        omp_set_num_threads(n_omp_threads);
        #pragma omp parallel for
        for (int i = 0; i < n_body; i++) {
            update_position(x, y, vx, vy, i);
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = t2 - t1;
        total += time_span;

        // printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

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

    printf("Elapsed time: %.5f\n", total.count());

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    
}


int main(int argc, char *argv[]){
    
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_omp_threads = atoi(argv[3]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("N Body Simulation Sequential Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    master();

    printf("Student ID: 119010545\n");
    printf("Name: Samuel Theofie\n");
    printf("Assignment 2: N Body Simulation OpenMP Implementation\n");
    
    return 0;

}
