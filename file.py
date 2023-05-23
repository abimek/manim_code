import numpy as np
from manim import *
from manim_physics import *

class Main(Scene):
    def construct(self):
        IntroAndStandardModel()
        ProtonsAreMadeOfParticles()


class IntroAndStandardModel(Scene):
    def construct(self):
        intro_text = Text('Quantum Field Theory and The Higgs Boson')
        by_who = Text('By: Abimalek Mekuriya')
        by_who.next_to(intro_text, DOWN)
        self.play(Write(intro_text, run_time=2), Write(by_who, run_time=2))
        self.wait(3)
        self.play(FadeOut(by_who))
        self.play(intro_text.animate.to_edge(UP))
        self.wait(2)
        standard_model = ImageMobject('assets/standard_model.jpg')
        standard_model.scale(.4)
        self.add(standard_model)
        self.wait(1)
        self.play(standard_model.animate.to_edge(LEFT))
        self.wait(1)
        standard_model_arrow = Arrow(start=np.array((1.0, 2.0, 0.0)), end=standard_model)
        standard_model_text = Text('The Standard Model')
        standard_model_text.scale(.5)
        standard_model_text.next_to(standard_model_arrow, RIGHT)
        self.play(Write(standard_model_arrow), Write(standard_model_text))
        self.wait(20)

class ProtonsAreMadeOfParticles(Scene):
    def construct(self):
        up = self.Quark(UP, RED)
        up_quark = VGroup(up[0], up[1], up[2])
        self.play(up_quark.animate.shift(LEFT*3))
        up2 = self.Quark(UP, RED)
        up2_quark = VGroup(up2[0], up2[1], up2[2])
        self.play(up2_quark.animate.shift(LEFT*2))
        down = self.Quark(DOWN, BLUE)
        down_quark = VGroup(down[0], down[1], down[2])
        self.play(down_quark.animate.shift(LEFT*2.5))

        proton = VGroup(up_quark, up2_quark, down_quark)
        circle = Circle(color=RED)
        circle.surround(proton)
        self.play(Create(circle))
        self.wait(1)
        proton_circle = Circle(fill_color=RED, fill_opacity=.8).scale(.6)
        proton_circle.move_to(proton.get_center())
        g = VGroup(circle, proton)
        self.play(Transform(g, proton_circle))
        neutron_circle = proton_circle.copy().set_fill(GRAY).set_color(GRAY)
        self.play(Create(neutron_circle), neutron_circle.animate.shift(RIGHT*2))


        circle2 = Circle(color=BLUE)
        circle2.scale(4)
        circle2.move_to(np.array((.5, 0, 0)))
        electron = Dot(color=BLUE).shift(RIGHT)
        self.play(GrowFromCenter(circle2))
        self.play(MoveAlongPath(electron, circle2), run_time=5, rate_functions=[linear])
        self.play(FadeOut(circle, circle2, proton_circle, neutron_circle, proton))

        self.play(electron.animate.shift(LEFT * 4 + np.array((-.5, 0, 0))))




    def Quark(self, arrowType, circleColor):
        up_arrow = Arrow(start=ORIGIN, end=arrowType)
        up_arrow.shift(RIGHT)
        self.play(Write(up_arrow))
        text = Text('Quark', color=BLUE)
        text.shift(RIGHT)
        text.next_to(up_arrow, RIGHT)
        self.play(Write(text))
        inner = VGroup(up_arrow, text)
        circle = Circle(color=circleColor)
        circle.surround(inner)
        self.play(Create(circle))

        temp_up = Group(circle, up_arrow, text)
        up_quark = temp_up.copy()
        up_quark.scale(.3)
        self.play(Transform(temp_up, up_quark))
        return [*circle, *inner, *text]

class AllAtomsAreTheSameAndVector(Scene):
    def construct(self):
        electron = Circle(radius=.05, fill_color=BLUE, color=BLUE, fill_opacity=1)
        electron.move_to(ORIGIN)
        self.add(electron)

        plus = Text('-')
        circle = Circle(fill_color=BLUE, fill_opacity=.8, color=BLUE)
        circle.surround(plus)
        electron_n = VGroup(plus, circle)
        self.play(Transform(electron, electron_n))
        self.wait(5)


        mars = Text('Electron From Mars')
        mars.scale(.3)
        electron_mars = electron_n.copy()
        self.play(electron_mars.animate.shift(LEFT*2))
        mars.next_to(electron_mars, UP)
        self.play(Write(mars))
        self.wait(2)
        electron_far_away = electron_n.copy()
        far_away = Text('In A Galaxy Far Far Away...')
        far_away.scale(.3)
        self.play(electron_far_away.animate.shift(RIGHT*2))
        far_away.next_to(electron_far_away, UP)
        self.play(Write(far_away))

        equal = Text('=')
        equal.move_to(LEFT)
        equal2 = Text('=')
        equal2.move_to(RIGHT)
        self.play(Write(equal))
        self.play(Write(equal2))
        self.wait(4)
        electron_mass = Text(r"m = 9.109e-31")
        electron_mass.to_edge(UP)
        electron_q =  Text(r"q = -1.602e-19")
        electron_q.next_to(electron_mass, DOWN)
        self.play(Write(electron_mass))
        self.play(Write(electron_q))
        self.wait(3)
        self.play(FadeOut(electron_n, electron_far_away, electron_mars, mars, far_away, equal, equal2, circle, plus, electron, electron_mass, electron_q))
        text = Text("Every Spot in Our Universe has a given coordinate, what better represents this that a...")
        text.scale(.5)
        text2 = Text("Vector Field", color=RED)
        text2.next_to(text, DOWN)
        self.play(Write(text))
        self.wait(1)
        self.play(Write(text2))
        self.wait(2)
        self.play(FadeOut(text, text2))
        populate_text = Text('We can populate vector fields with many different values such as...')
        populate_text.scale(.5)
        scalars = Text('Scalars such as 1, 2, 3 (Higgs Boson Hint Hint..)')
        scalar_example = Circle(radius=.2, fill_color=BLUE, fill_opacity=1.0, color=BLUE)
        scalars.scale(.5)
        scalar_example.next_to(scalars, RIGHT)
        vectors = Text('Vectors such as charge')
        vector_example = Arrow(start=ORIGIN, end=UP+RIGHT)
        vector_example.next_to(vectors, RIGHT)
        vector_example.shift(DOWN*2 + LEFT*3)
        vectors.scale(.5)
        scalars.next_to(populate_text, DOWN)
        vectors.next_to(scalars, DOWN)
        self.play(populate_text.animate.shift(UP*2))
        self.play(Write(populate_text))
        self.wait(3)
        self.play(Write(scalars), Create(scalar_example))
        self.wait(5)
        self.play(Write(vectors), Write(vector_example))
        self.wait(4)
        self.play(FadeOut(populate_text, scalars, vectors, scalar_example, vector_example))


class VectorField2D(Scene):

    def construct(self):
        plane = NumberPlane()
        self.play(GrowFromCenter(plane))

        scalars = Text('Scalars', color=RED)
        scalars.to_edge(UP)
        self.play(Write(scalars))
        self.wait(3)

        points = [
            x*RIGHT+y*UP
            for x in np.arange(-10, 10, 1)
            for y in np.arange(-10, 10, 1)
        ]

        point_field = []
        for p in points:
            d = Dot()
            d.shift(p)
            point_field.append(d)

        point_g = VGroup(*point_field)
        self.play(Create(point_g))

        self.wait(4)

        vector_field = []
        for p in points:
            f = 0.5 * RIGHT + 0.5 * UP
            vec = Vector(f).shift(p)
            vector_field.append(vec)

        vector_g = VGroup(*vector_field)
        vectorsd = Text('Vectors', color=RED)
        vectorsd.to_edge(UP)
        self.play(Transform(scalars, vectorsd))
        self.play(FadeOut(point_g), Create(vector_g))
        self.wait(4)
        rotate_animations = []
        for i, v in enumerate(vector_field):
            rotate_animations.append(Rotate(v, PI/2, about_point=points[i]))
        self.play(*rotate_animations)
        self.wait(8)

        electron_field = []
        for p in points:
            plus = Text('-')
            circle = Circle(fill_color=BLUE, fill_opacity=.8, color=BLUE)
            circle.surround(plus)
            electron_n = VGroup(plus, circle)
            electron_n.shift(p)
            electron_field.append(electron_n)

        electron_g = VGroup(*electron_field)
        electext = Text('Electrons', color=RED)
        electext.to_edge(UP)
        self.play(FadeOut(scalars), Create(electext))
        self.play(FadeOut(vector_g), Create(electron_g))
        self.wait(4)


class ElectronsAreJustOneField(ThreeDScene):

    def construct(self):
        self.set_camera_orientation(0, -np.pi/2)
        plane = NumberPlane()
        self.play(GrowFromCenter(plane))

        electron = Text('Electrons', color=RED)
        electron.to_edge(UP)
        self.add(electron)

        points = [
            x*RIGHT+y*UP
            for x in np.arange(-7, 7, 1)
            for y in np.arange(-4, 4, 1)
        ]
        electron_field = []
        for p in points:
            plus = Text('-')
            circle = Circle(fill_color=BLUE, fill_opacity=.8, color=BLUE)
            circle.surround(plus)
            electron_n = VGroup(plus, circle)
            electron_n.shift(p)
            electron_field.append(electron_n)
        electron_g = VGroup(*electron_field)
        self.play(Create(electron_g))

        self.wait(1)
        self.play(FadeOut(electron))
        p_f_p = Text('Let the letter p stand for the potential to have an electron at that position')
        p_f_p.to_edge(UP)
        p_f_p.scale(.4)
        self.play(Write(p_f_p))
        self.wait(1)
        p_for_potential = Text('p').scale(.5)
        p_field = []
        for p in points:
            p_field.append(p_for_potential.copy().shift(p))
        p_g = VGroup(*p_field)
        self.wait(6)
        self.play(FadeOut(electron_g), Create(p_g))
        self.wait(4)
        keep_one = Text('Lefts keep one electron as that electron has reached the potential value for it to exist')
        keep_one.scale(.4)
        keep_one.to_edge(UP)
        self.play(FadeOut(p_f_p))
        self.play(Write(keep_one))
        plus = Text('-')
        circle = Circle(fill_color=BLUE, fill_opacity=.8, color=BLUE)
        circle.surround(plus)
        electron_n = VGroup(plus, circle)
        electron_n.scale(1.5)
        electron_n.shift(UP+RIGHT)
        self.play(Create(electron_n))
        self.wait(2)
        # first paramater is the angle that way
        self.move_camera((.8*np.pi)/2, -np.pi/2, zoom=.1)
        self.wait(20)
        main_field = VGroup(electron_n, keep_one, plane, p_g, )
        fields = [main_field]
        particles = []
        fields_l = [{"symbol": "γ", "name": "photon", "color": RED}, {"symbol": "Uf", "name": "gluon", "color": YELLOW}]
        for i,f in enumerate(fields_l):
            plane2 = NumberPlane()
            self.play(GrowFromCenter(plane2))
            particle = Text(f["name"], color=f["color"])
            particle.to_edge(UP)
            self.add(particle)
            p_field_2 = []
            for p in points:
                p_field_2.append(p_for_potential.copy().shift(p))
            p_g_2 = VGroup(*p_field_2)
            self.play(Create(p_g_2))
            pho = Text(f["symbol"])
            circle = Circle(fill_color=BLUE, fill_opacity=.8, color=f["color"])
            circle.surround(pho)
            particle_n = VGroup(plus, circle)
            particles.append(particle_n)
            particle_n.scale(1.5)
            particle_n.shift(UP+RIGHT)
            self.play(Create(particle_n))
            field_2_g = VGroup(plane2, particle, pho, particle_n, p_g_2)
            self.play(field_2_g.animate.shift(OUT*3*(i+1)))
            self.wait(2)
            fields.append(field_2_g)

        interactions_text = Text('These different particle fields have the ability to interact, causing an excitement in one')
        interactions_text.scale(.5)
        interactions_text.shift(DOWN*3)
        interactions_text_2 = Text('to disappear and an excitement of the same energy to appear in another field')
        interactions_text_2.scale(.5)
        interactions_text_2.next_to(interactions_text, DOWN)
        self.move_camera((.7*np.pi)/2, zoom=.6)
        self.wait(2)
        self.add_fixed_in_frame_mobjects(interactions_text)
        self.add_fixed_in_frame_mobjects(interactions_text_2)
        self.wait(5)
        self.play(particles[0].animate.shift(OUT*(-3)))
        self.wait(3)
        for i,f in enumerate(fields):
            self.play(f.animate.shift(-OUT*3*(i+1)))
        self.move_camera(0, 0, zoom=1)
        self.wait(5)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 42
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-1.5, +1.5],
            u_range=[-1.5, +1.5]
        )

        axes = ThreeDAxes()
        gauss_plane.scale(2, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        gauss_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (GREEN, 0.8)], axis=2)

        grid = NumberPlane(x_range=[-10, 10, 2], y_range=[-5, 5, 2], x_length=axes.x_length, y_length=axes.y_length)

        self.play(Create(grid), Create(gauss_plane))
        self.wait(10)
        grid_2 = NumberPlane(x_range=[-60, 60, 2], y_range=[-60, 60, 2], x_length=40, y_length=40)
        self.play(Transform(grid, grid_2))
        self.wait(7)
        self.play(gauss_plane.animate.shift(LEFT*13))
        self.wait(1)
        self.play(gauss_plane.animate.shift(UP*13))
        self.wait(5)
        why_electron_clouds = Text("The field extends out forever and large waves are our REAL particles")
        why_electron_clouds.scale(.5)
        why_electron_clouds.shift(DOWN*2)
        why_electron_clouds_2 = Text("while everywhere else we have small waves which are our virtual particles")
        why_electron_clouds_2.scale(.5)
        why_electron_clouds_2.next_to(why_electron_clouds, DOWN)
        self.add_fixed_in_frame_mobjects(why_electron_clouds, why_electron_clouds_2)
        self.wait(8)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

class HiggsIntro(Scene):

    def construct(self):
        what_about_higgs = Text("What about the higgs field...", color=BLUE)
        self.play(Write(what_about_higgs))
        self.wait(1)
        self.play(what_about_higgs.animate.to_edge(UP))
        intrinsic_mass = Text("Most atoms have an associated intrinsic mass")
        self.play(Write(intrinsic_mass))
        self.play(intrinsic_mass.animate.move_to(what_about_higgs.get_center() + DOWN))
        t2 = Table(
            [["Electron", "9.109e-31 kg"],
             ["photon", "0 :o"],
             ["up quark", "3.56e-30 kg"],
             ["muon", "1.883e-28 kg"]
             ],
            row_labels=[Text("1"), Text("2"), Text("3"), Text("4")],
            col_labels=[Text("Particle"), Text("Mass", color=RED)],
            include_outer_lines=True,
            arrange_in_grid_config={"cell_alignment": RIGHT})
        t2.scale(.7)
        t2.shift(DOWN)
        self.play(Create(t2))
        #coords = [(2, 2), (2, 3), (2, 4)]
        self.wait(8)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        higgs_field_gives_mass = Text("The higgs field gives them their mass", color=BLUE)
        higgs_field_gives_mass.scale(.7)
        self.play(Write(higgs_field_gives_mass))
        self.wait(2)
        self.play(higgs_field_gives_mass.animate.shift(UP))

        example_text = Text("Let's take an example", color=BLUE)
        self.play(Write(example_text))
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        plane = NumberPlane()
        self.play(GrowFromCenter(plane))

        scalars = Text('Higgs Field', color=RED)
        scalars.to_edge(UP)
        self.play(Write(scalars))
        self.wait(2)

        points = [
            x*RIGHT+y*UP
            for x in np.arange(-10, 10, 1)
            for y in np.arange(-10, 10, 1)
        ]

        point_field = []
        for p in points:
            d = Dot()
            d.shift(p)
            point_field.append(d)

        point_g = VGroup(*point_field)
        self.play(Create(point_g))
        self.wait(4)
        plus = Text('-')
        circle = Circle(fill_color=BLUE, fill_opacity=.8, color=BLUE)
        circle.surround(plus)
        electron_n = VGroup(plus, circle)
        electron_n.scale(1.5)
        electron_n.to_edge(LEFT)
        self.play(Create(electron_n))
        self.wait(2)
        self.play(electron_n.animate(run_time=5).shift(RIGHT*10))
        self.wait(3)
        pho = Text('γ')
        circle = Circle(fill_color=YELLOW, fill_opacity=.8, color=YELLOW)
        circle.surround(pho)
        photon_n = VGroup(pho, circle)
        photon_n.to_edge(LEFT)
        photon_n.shift(UP)
        self.play(FadeOut(point_g))
        self.wait(5)
        self.play(Create(photon_n))
        self.wait(2)
        self.play(photon_n.animate(run_time=2).shift(RIGHT*10))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class HiggsBoson(Scene):
    def construct(self):

        hig = Text('H', color=WHITE)
        circle = Circle(fill_color=RED, fill_opacity=.3, color=RED)
        circle.surround(hig)
        higgs_n = VGroup(hig, circle)
        higgs_n.to_edge(LEFT)
        higgs_n.scale(.5)
        self.play(Create(higgs_n))
        self.wait(4)
        self.play(higgs_n.animate().to_edge(UP))
        self.wait(3)
        tracker = ValueTracker(0)

        sin_graph_l = lambda x: (3*((1/(1+np.power(np.e, x)))*(1-(1/(1+np.power(np.e, x))))) * np.cos(4*x+tracker.get_value()))
        sine_graph = always_redraw(lambda: FunctionGraph(
            sin_graph_l,
            color=BLUE
        ))

        self.play(Create(sine_graph))
        self.wait(2)
        self.play(tracker.animate(run_time=12).set_value(80))
        self.wait(4)

        self.play(FadeOut(sine_graph), higgs_n.animate.move_to(ORIGIN))
        self.wait(2)
        self.play(higgs_n.animate.scale(1))
        self.wait(1)
        tracker2 = ValueTracker(0)
        sin_graph_l2 = lambda x: (3*((1/(1+np.power(np.e, x)))*(1-(1/(1+np.power(np.e, x))))) * np.cos(4*x+tracker2.get_value()))
        sine_graph_n = always_redraw(lambda: FunctionGraph(
            sin_graph_l2,
            color=RED
        ))
        tracker.set_value(0)
        self.play(ReplacementTransform(higgs_n, sine_graph_n))
        self.play(tracker2.animate(run_time=7).set_value(70))
        self.wait(2)






class HiggsLevel(MovingCameraScene):

    def construct(self):
         axes = Axes(
             x_range=[-10, 10, 1],
             y_range=[-7, 7, 1],
             axis_config={"color": WHITE},
             y_axis_config={
                 "numbers_to_include": np.array((1, 2, 3)),
                 "numbers_with_elongated_ticks": np.array((1, 2, 3))
             },
             tips=False
         )

         sin_graph_2 = axes.plot(lambda x: (.1*np.sin(20*x)+1), color=RED, x_range=[0, 10, .001])
         higgs_label_2 = axes.get_graph_label(sin_graph_2, label="Higgs Energy Level")
         higgs_label_2.scale(.5)
         higgs_label_2.shift(UP + UP/2)
         vert_line = axes.get_vertical_line(
             axes.i2gp(4, sin_graph_2), color=YELLOW, line_func=Line
         )
         self.play(Create(axes, run_time=3))
         self.wait(6)
         self.play(Create(sin_graph_2, runtime=2), Create(higgs_label_2, runtime=2))
         self.wait(6)

         sin_graph = axes.plot(lambda x: (.1*np.sin(20*x)), color=BLUE, x_range=[0, 10, .001])
         higgs_label = axes.get_graph_label(sin_graph, label="Any Other Field Energy Level")
         higgs_label.scale(.5)
         higgs_label.shift(DOWN)
         self.play(Create(sin_graph), Create(higgs_label))
         self.wait(5)
         self.play(Create(vert_line))
         self.wait(3)
         self.play(self.camera.frame.animate.scale(.5))
         self.wait(4)

         arr = Arrow(start=ORIGIN + UP+RIGHT*2, end=vert_line, color=YELLOW)
         arr.scale(.9)
         e_val = Text("Vacuum Expectation Value of 246 GeV")
         e_val.scale(.2)
         e_val.next_to(arr, UP)
         self.play(Create(arr), Create(e_val))
         self.wait(4)
         self.play(*[FadeOut(mob) for mob in self.mobjects])
         self.wait(1)


class Final(Scene):
    def construct(self):

        b = Text("Quantum Field Theory is our current best understand of the universe as it explains ", color=BLUE)
        b.scale(.5)
        b1 = Text("everything at the quantum level while following the laws of relativity", color=BLUE)
        b1.scale(.5)
        b1.next_to(b, DOWN)
        self.play(Write(b, run_time=3))
        self.play(Write(b1, run_time=2))
        self.wait(3)
        self.play(b.animate.to_edge(UP), b1.animate.shift(UP*3+UP/3))
        self.wait(2)
        self.play(FadeOut(b), FadeOut(b1))
        but_gravity = Text("Well, almost everything, except gravity, but they hypothesize the existance of a graviton", color=BLUE)
        but_gravity.scale(.5)
        self.wait(1)
        self.play(Write(but_gravity))
        self.wait(20)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        text = Text("Quantum Field Theory", color=RED)
        self.play(Create(text))
        self.wait(5)
