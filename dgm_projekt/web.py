import model
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, column, row
from bokeh.models import ColumnDataSource, Button, Arrow, NormalHead, Div, Range1d, Spinner
from bokeh.events import ButtonClick

config = {
    "epochs": 350,
    "learning_rate": 0.2,
    "init_theta": 1,
    "init_psi": 1,
    "plot_interval": (-2, 2),
    "update_interval": 50 
}

def quiver(self, X, Y, U, V, color, alpha=1):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cx, cy, cu, cv = X[i, j], Y[i, j], U[i, j], V[i, j]
            d = np.sqrt(cu**2 + cv**2) 
            head_size = min(4, 4 * (d/0.1))
            nh = NormalHead(fill_color=color, fill_alpha=alpha, line_color=color, size=head_size)
            self.add_layout(Arrow(end=nh, line_color=color, line_width=2, line_alpha=alpha, level="underlay",
                   x_start=cx, y_start=cy, x_end=cx+cu, y_end=cy+cv))

def train():
    # standard gan:
    gan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])

    # NGAN:
    ngan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    ngan.set_loss('NGAN')

    # WGAN:
    wgan = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    wgan.set_loss('WGAN')

    # WGAN-GP:
    wgan_gp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    wgan_gp.set_loss('WGAN')
    wgan_gp.set_regularization_loss('WGP')

    # IGP:
    igp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    igp.set_instance_noise(True)

    # GP:
    gp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    gp.set_regularization_loss('GP')

    # CRGP:
    crgp = model.Model(iterations=config["epochs"], learning_rate=config["learning_rate"])
    crgp.set_regularization_loss('CRGP')

    # DRAGAN:
    dragan = model.Model()
    dragan.set_regularization_loss('DRAGAN')
    dragan.set_instance_noise(True)

    # Simple LeCam_Regularization:
    simple_lecam_reg = model.Model()
    simple_lecam_reg.set_regularization_loss('SimpleLeCam')

    # LeCam as in paper: "Regularizing Generative Adversarial Networks under Limited Data"
    lecam_reg = model.Model()
    lecam_reg.set_regularization_loss('LeCam')

    training_algorithms = [
        (gan, "Standard GAN"),
        (ngan, "Non-saturating GAN"),
        (wgan, "Wasserstein GAN"),
        (wgan_gp, "Wasserstein GAN with GP"),
        (igp, "Instance Noise GAN"),
        (gp, "GAN with Gradient Penalty"),
        (crgp, "Critically Penalized GAN"),
        (dragan, "DRAGAN Gradient Penalty"),
        (simple_lecam_reg, "LeCam-Distance as regularization"),
        (lecam_reg, "lecamgan")
    ]
    
    plots = []
    sources = {}
    start_sources = []
    for gan_model, name in training_algorithms:
        thetas, psis = gan_model.train(init_theta=config["init_theta"], init_psi=config["init_psi"])
        X, Y, U, V = gan_model.get_vectors(interval=config["plot_interval"], steps=15)
        rmin, rmax = config["plot_interval"]
        x_range = Range1d(rmin, rmax, bounds=(rmin, rmax))
        y_range = Range1d(rmin, rmax, bounds=(rmin, rmax))
        p = figure(title=name, x_axis_label='θ', y_axis_label='Ψ', tools="pan,wheel_zoom,box_zoom,reset,save", 
                x_range=x_range, y_range=y_range)
        #quiver(p, X, Y, U, V, color="black")
        # ^^^ use this for more control (but it's slower)

        cds = ColumnDataSource(data=dict(x_start=X, y_start=Y, x_end=X+U, y_end=Y+V))
        p.add_layout(Arrow(end=NormalHead(line_color="black", fill_color="black", size=4), source=cds, 
                        x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end', 
                        line_width=2, level="underlay", line_color="black"))

        source = ColumnDataSource(data=dict(x=[], y=[]))
        source_start = ColumnDataSource(data=dict(x=[], y=[]))
        p.scatter("x", "y", source=source, color="blue", size=8)
        p.scatter("x", "y", source=source_start, color="red", size=8)

        if name == "Wasserstein GAN":
            clamp = gan_model.clamp
            x_shade = [rmin, rmax, rmax, rmin]
            y_shade = [rmax, rmax, clamp, clamp]
            p.patch(x_shade, y_shade, color="grey", alpha=0.5, line_width=0)
            x_shade2 = [rmin, rmax, rmax, rmin]
            y_shade2 = [-clamp, -clamp, rmin, rmin]
            p.patch(x_shade2, y_shade2, color="grey", alpha=0.5, line_width=0)

            source_start.stream(dict(x=[config["init_theta"]], y=[config["init_psi"]]), rollover=1)
        else:
            source_start.stream(dict(x=[config["init_theta"]], y=[config["init_psi"]]), rollover=1)

        sources[name] = (source, thetas, psis)
        start_sources.append(source_start)
        plots.append(p) 

    return plots, sources, start_sources

def ui(plots, sources, start_sources):
    frame = 0
    paused = False 
    def get_update(sources, start_sources):
        def update():
            nonlocal frame, paused 
            if paused: return 
            reset = False 
            if frame >= config["epochs"]:
                reset = True 
                frame = 0 
                
            for source in start_sources:
                source.stream(dict(x=[config["init_theta"]], y=[config["init_psi"]]), rollover=1)

            for source, thetas, psis in sources.values():
                if reset:
                    source.data = dict(x=[], y=[])
                source.stream(dict(x=[thetas[frame]], y=[psis[frame]]), rollover=config["epochs"])
            frame += 1 
        return update 

    grid = gridplot(plots, ncols=3, width=300, height=300, toolbar_options=dict(logo=None))
    frame = 0
    curdoc().add_periodic_callback(get_update(sources, start_sources), config["update_interval"])

    # Buttons & final layout 
    header = Div(text="<h1>DiracGAN</h1>")
    footer = Div(text="<h4 style='color:gray'>Deep Generative Models 2024 - Project by Emanuele Harrasser, Milan Binz, Julian Bayer</h4>")

    anim_controls = Div(text="<h2>Animation Controls</h2>")
    reset_btn = Button(label="Reset Animation")
    pause_btn = Button(label="Pause Animation")
    skip_to_end_btn = Button(label="Skip to end") 

    def reset(sources):
        def reset_action(_):
            nonlocal frame 
            frame = 0
            for source, _, _ in sources.values():
                source.data = dict(x=[], y=[])
        return reset_action

    def toggle_paused(): 
        def pause_action(_):
            pause_btn.label = ["Pause Animation", "Resume Animation"][["Resume Animation", "Pause Animation"].index(pause_btn.label)]
            nonlocal paused
            paused = not paused
        return pause_action

    def skip_to_end(sources):
        def skip_to_end_action(_):
            nonlocal frame 
            if not paused: toggle_paused()(_) 
            for source, thetas, psis in sources.values():
                source.data = dict(x=thetas, y=psis)
                frame = config["epochs"]
        return skip_to_end_action

    reset_btn.on_event(ButtonClick, reset(sources))
    pause_btn.on_event(ButtonClick, toggle_paused())
    skip_to_end_btn.on_event(ButtonClick, skip_to_end(sources))
    anim_buttons = row(reset_btn, pause_btn, skip_to_end_btn)

    param_controls = Div(text="<h2>Parameters</h2>")
    retrain_button = Button(label="Retrain") 
    spinner_epochs = Spinner(title="Epochs", low=1, high=10000, step=1, value=config["epochs"])
    spinner_lr = Spinner(title="Learning Rate", low=1e-6, high=1, step=0.01, value=config["learning_rate"])
    rmin, rmax = config["plot_interval"]
    spinner_start_x = Spinner(title="Initial θ", low=rmin, high=rmax, step=0.01, value=config["init_theta"])
    spinner_start_y = Spinner(title="Initital Ψ", low=rmin, high=rmax, step=0.01, value=config["init_psi"])

    def retrain(): 
        def retrain_action(_):
            nonlocal sources 
            config["epochs"] = spinner_epochs.value 
            config["learning_rate"] = spinner_lr.value 
            config["init_theta"] = spinner_start_x.value             
            config["init_psi"] = spinner_start_y.value             
            _, new_sources, _ = train() 
            for name, _ in sources.items():
                sources[name] = (sources[name][0], new_sources[name][1], new_sources[name][2])
            reset(sources)(_) 
        return retrain_action
    
    retrain_button.on_event(ButtonClick, retrain()) 
    param_buttons = column(column(row(spinner_epochs, spinner_lr), row(spinner_start_x, spinner_start_y)), retrain_button) 

    layout = column(header, grid, anim_controls, anim_buttons, param_controls, param_buttons, footer)
    return layout 

def setup():
    plots, sources, start_sources = train() 
    root = ui(plots, sources, start_sources)
    curdoc().add_root(root) 
    curdoc().set_title("DiracGAN")

setup() 