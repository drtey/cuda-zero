#ifndef __CPU_ANIM_H__
#define __CPU_ANIM_H__

#include <GL/glut.h> // Asegúrate de que las bibliotecas OpenGL estén instaladas
#include <iostream>

struct CPUAnimBitmap {
    unsigned char* pixels;
    int width, height;
    void* dataBlock;
    void (*fAnim)(void*, int);
    void (*animExit)(void*);
    void (*clickDrag)(void*, int, int, int, int);
    int dragStartX, dragStartY;

    CPUAnimBitmap(int w, int h, void* d = NULL) {
        width = w;
        height = h;
        pixels = new unsigned char[width * height * 4];
        dataBlock = d;
        clickDrag = NULL;
    }

    ~CPUAnimBitmap() {
        delete[] pixels;
    }

    unsigned char* get_ptr(void) const { return pixels; }
    long image_size(void) const { return width * height * 4; }

    void click_drag(void (*f)(void*, int, int, int, int)) {
        clickDrag = f;
    }

    void anim_and_exit(void (*f)(void*, int), void (*e)(void*)) {
        CPUAnimBitmap** bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;
        int c = 1;
        const char* dummy = ""; // Corregido el uso de const char*
        glutInit(&c, const_cast<char**>(&dummy));
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize(width, height);
        glutCreateWindow("bitmap");
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        if (clickDrag != NULL)
            glutMouseFunc(mouse_func);
        glutIdleFunc(idle_func);
        glutMainLoop();
    }

    static CPUAnimBitmap** get_bitmap_ptr(void) {
        static CPUAnimBitmap* gBitmap;
        return &gBitmap;
    }

    static void mouse_func(int button, int state, int mx, int my) {
        if (button == GLUT_LEFT_BUTTON) {
            CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                bitmap->clickDrag(bitmap->dataBlock, bitmap->dragStartX, bitmap->dragStartY, mx, my);
            }
        }
    }

    static void idle_func(void) {
        static int ticks = 1;
        CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
        bitmap->fAnim(bitmap->dataBlock, ticks++);
        glutPostRedisplay();
    }

    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27: // Escape key
                CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
                bitmap->animExit(bitmap->dataBlock);
                exit(0);
        }
    }

    static void Draw(void) {
        CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
        glutSwapBuffers();
    }
};

#endif // __CPU_ANIM_H__
