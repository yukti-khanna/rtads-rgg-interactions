"""Fixed physical-size, editable panel exports for the final figure package."""
import pickle
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.text import Text, Annotation
from matplotlib.font_manager import FontProperties, findfont

WIDTH_CM, HEIGHT_CM, FONT_PT = 9, 5, 7

def set_panel_fonts(fig):
    # Fail explicitly rather than silently substituting a different font.
    findfont(FontProperties(family='Arial'), fallback_to_default=False)
    for text in fig.findobj(Text):
        text.set_text(text.get_text().replace('log₁₀(−B₂₂)',r'$\log_{10}(-B_{22})$'))
        text.set_fontfamily('Arial')
        text.set_fontsize(FONT_PT)

def export_panels(fig, name, destination):
    """Clone each axes so resizing never changes the assembled overview."""
    directory=Path(destination)/'panels_9x5cm'
    directory.mkdir(parents=True,exist_ok=True)
    serialized=pickle.dumps(fig)
    for index in range(len(fig.axes)):
        panel=pickle.loads(serialized)
        ax=panel.axes[index]
        for other in list(panel.axes):
            if other is not ax: panel.delaxes(other)
        panel.set_layout_engine(None)
        panel.set_size_inches(WIDTH_CM/2.54,HEIGHT_CM/2.54,forward=True)
        # Figure-level legends in the LEF1 composite apply only to its energy/contact panels.
        for legend in list(panel.legends):
            if index<3: legend.remove()
            else:
                handles=legend.legend_handles
                labels=[text.get_text() for text in legend.get_texts()]
                legend.remove()
                panel.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,1),ncol=2,frameon=False)
        ax.set_position([.22,.25,.72,.52])
        # Standalone panels use their figure caption for the descriptive title.
        # Reserving the top strip for legends prevents title/legend collisions.
        ax.set_title('')
        for getter,setter in [(ax.get_xlabel,ax.set_xlabel),(ax.get_ylabel,ax.set_ylabel)]:
            setter(getter().replace(' simulated score',' simulated').replace('Frozen predicted score','Predicted'))
        if 'shuffle_and_LEF1' in name and index==1:
            handles,labels=ax.get_legend_handles_labels()
            ax.legend(handles,['Kd/ND','Kd+LB/ND','LB/ND'],loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,columnspacing=.7,handlelength=1.2)
        if 'shuffle_and_LEF1' in name and index>=3:
            ax.tick_params(axis='x',rotation=0)
        if 'shuffle_and_LEF1' in name and index==2:
            # Outcomes are written beside every marker; the repeated legend
            # competes with the two column labels at this publication size.
            if ax.get_legend() is not None:ax.get_legend().remove()
            for collection in ax.collections:
                if hasattr(collection,'set_sizes'):collection.set_sizes([16])
            ax.set_ylim(len(ax.get_yticks())-.25,-.75)
        for text in list(ax.texts):
            if text.get_text() in list('ABCDEFG'):
                text.remove()
            elif text.get_transform()==ax.transAxes and text.get_position()[1]<0:
                # Statistical method notes are already defined in the caption.
                text.remove()
        panel.text(.025,.96,chr(65+index),va='top',weight='bold')
        set_panel_fonts(panel)
        identity=[(frame,column) for target,frame,column in getattr(panel,'_identity_panels',[]) if target is ax]
        if identity:
            for text in list(ax.texts):
                if isinstance(text,Annotation):text.remove()
        # Fit decorations into the fixed canvas without bbox_inches='tight',
        # which would change the requested PDF/SVG dimensions.
        for _ in range(8):
            panel.canvas.draw()
            box=ax.get_tightbbox(panel.canvas.get_renderer()).transformed(panel.transFigure.inverted())
            pos=ax.get_position()
            left=max(0,.02-box.x0);right=max(0,box.x1-.98)
            bottom=max(0,.03-box.y0);top=max(0,box.y1-.97)
            if max(left,right,bottom,top)<.002:break
            width=pos.width-left-right;height=pos.height-bottom-top
            if width<.20 or height<.20:break
            ax.set_position([pos.x0+left,pos.y0+bottom,width,height])
        if identity:
            from plot_master_validation import identity_labels
            frame,column=identity[0]
            identity_labels(panel,ax,frame,column,'pKd',yerr='pKd_error')
            set_panel_fonts(panel)
        stem=f'{name}_panel_{chr(65+index)}'
        for extension in ['pdf','svg','png']:
            panel.savefig(directory/f'{stem}.{extension}',dpi=300,bbox_inches=None)
        mappings={
            'FigureS10_simulation_and_intensity':['S8G','S8H','S8A','S8D','S8F'],
            'FigureS11_shuffle_and_LEF1_context':['S8B','S8C','S9A','S9B','S9C','S9D']}
        if name in mappings:
            target=mappings[name][index]
            for text in panel.texts:
                if text.get_text()==chr(65+index):text.set_text(target[-1])
            manuscript=Path(destination)/'manuscript_panels'
            manuscript.mkdir(parents=True,exist_ok=True)
            for extension in ['pdf','svg','png']:
                panel.savefig(manuscript/f'Figure{target}.{extension}',dpi=300,bbox_inches=None)
        plt.close(panel)
    (directory/'README.txt').write_text(
        'Each PDF/SVG canvas is exactly 9 cm wide by 5 cm high. All text uses Arial 7 pt.\n'
        'PNG uses 300 dpi (approximately 1062 by 590 pixels). Overview composites are separate.\n'
        'Panel A of Figure 5 remains an author-supplied NMR placeholder.\n')
