# sphinx_drafts: a sphinx extension to mark pages as draft
# and automatically mark referring pages as drafts
#
# Copyright (C) 2012 Diego Veralli <diegoveralli@yahoo.co.uk>
#
#  This file is part of sphinx_drafts.
#
#  sphinx_drafts is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sphinx_drafts is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with sphinx_drafts. If not, see <http://www.gnu.org/licenses/>.
#
"""a sphinx extension to mark pages as draft and automatically mark
referring pages as drafts
"""

import sphinx
from docutils import nodes
from docutils.parsers.rst import Directive

DRAFT_DOCS_TEXT = "This is draft documentation"


class refdoc_marker(nodes.General, nodes.Element):
    def __init__(self, target_doc):
        super(refdoc_marker, self).__init__()
        self.target_doc = target_doc


class draft_marker(nodes.General, nodes.Element):
    def __init__(self, check):
        super(draft_marker, self).__init__()
        self.check = check


class DraftNote(Directive):
    """Usage: .. draft:: 'yes' or .. draft:: 'check'
    """
    has_content = True
    required_arguments = 1

    def run(self):
        if self.arguments[0] == 'yes':
            check = False
        elif self.arguments[0] == 'check':
            check = True
        else:
            msg = 'Argument must be "yes" or "check", found "%s" instead' % \
                self.arguments[0]
            raise Exception(msg)

        marker = draft_marker(check)
        self.state.nested_parse(self.content, self.content_offset,
                                marker, match_titles=1)
        return [marker]


class DraftInfo(object):
    def __init__(self, status=None, link_references=None, \
                     draft_dependencies=None):
        self.status = status
        self.link_references = link_references
        self.draft_dependencies = draft_dependencies


def get_draft_info(app, docname, doctree):
    """Find draft info either in the env cache or the doctree itself.
    We can't get the doc name from the doctree in the doctree-read
    hook so we initially store it with the doc itself and then copy it
    to the env cache as soon as we get a name.
    FIXME: Can we really not get the document name in doctree-read?
    """
    env = app.builder.env
    if not hasattr(env, 'draft_doc_status'):
        env.draft_doc_status = {}

    if docname == None or not docname in env.draft_doc_status:
        retval = doctree.attributes.get('draft_info')
        if retval == None:
            retval = DraftInfo()
            doctree.attributes['draft_info'] = retval

        if docname != None:
            env.draft_doc_status[docname] = retval
    else:
        retval = env.draft_doc_status[docname]

    return retval


def process_draft_markers(app, doctree):
    """This is called in the doctree-read hook, it sets the draft status
    when it's declared statically (ie, == 'yes'), and it also caches the
    foreign doctree references of any links in the doctree.

    FIXME: Can we grab link references in doctree-resolved instead? Maybe
    that'd save us from having to build the absolute path of the referenced
    documents ourselves..
    """

    #No name available, or none that I could find, so pass None
    draft_info = get_draft_info(app, None, doctree)

    for node in doctree.traverse(draft_marker):
        curr = draft_info.status
        if curr == None or curr == 'check':
            draft_info.status = 'check' if node.check else 'yes'

    for node in doctree.traverse(sphinx.addnodes.pending_xref):
        if 'reftarget' in node.attributes:
            reftarget = node.attributes['reftarget']
            marker = refdoc_marker(reftarget)
            doctree.append(marker)


def locate_relative_doc(refdoc_name, doc_name):
    """Converts a relative doc reference to an absolute reference
    (within the source tree).

    FIXME: This is broken for a bunch of cases, there's probably
    a builtin sphinx function that we should be using, if there
    isn't then TODO implement properly.
    """

    if doc_name.startswith('/'):
        return doc_name
    elif '/' in refdoc_name:
        split_point = refdoc_name.rindex('/')
        return refdoc_name[:split_point + 1] + doc_name

    return doc_name


def find_doctree(app, referencing_docname, docname):
    env = app.builder.env
    name = locate_relative_doc(referencing_docname, docname)
    return (name, env.get_doctree(name))


def update_link_references(doctree, draft_info):
    refs = draft_info.link_references
    if refs == None:
        refs = []
        draft_info.link_references = refs

    for node in doctree.traverse(refdoc_marker):
        if node.target_doc not in refs:
            refs.append(node.target_doc)


def update_status(app, doctree, docname, seen_docs):
    """Returns the draft status and draft dependencies (if there were)
    of the doctree, recursively evaluating the status of any foreign
    doctree references if necessary.

    The status of any referenced document that is evaluated will be
    stored in the env cache.
    """

    draft_info = get_draft_info(app, docname, doctree)
    curr = draft_info.status

    #The was no draft directive on the page, this is not a draft
    if curr == None:
        return ('no', None)
    #We have an answer for this doc already, either way
    if curr != 'check':
        return (curr, draft_info.draft_dependencies)

    seen_docs.append(docname)

    update_link_references(doctree, draft_info)

    if not draft_info.link_references:
        return (curr, draft_info.link_references)

    draft_dependencies = []
    for rel_depname in draft_info.link_references:
        depname, dep_doctree = find_doctree(app, docname, rel_depname)
        dep_info = get_draft_info(app, depname, dep_doctree)

        if dep_info.status == 'yes':
            draft_dependencies.append(depname)

        if dep_info.status == 'check' and depname not in seen_docs:
            status, dependencies = update_status(app, dep_doctree, depname,
                                                 seen_docs)
            dep_info.status = status
            dep_info.draft_dependencies = dependencies

    if len(draft_dependencies) > 0:
        return ('yes', draft_dependencies)
    else:
        return (curr, draft_info.draft_dependencies)


def create_draft_warning(draft_dependencies=None):
    text = DRAFT_DOCS_TEXT
    if draft_dependencies:
        text += " because it links to the following draft pages:"
    t = nodes.Text(text)
    p = nodes.paragraph()
    p.append(t)

    warning = nodes.warning()
    warning.append(p)
    if draft_dependencies:
        lst = nodes.bullet_list()
        for dep in draft_dependencies:
            item = nodes.list_item()
            item_p = nodes.paragraph()
            item_t = nodes.Text(dep)
            item_p.append(item_t)
            item.append(item_p)
            lst.append(item)

        warning.append(lst)
    return warning


def process_draft_nodes_resolved(app, doctree, docname):
    draft_info = get_draft_info(app, docname, doctree)

    for node in doctree.traverse(draft_marker):
        if draft_info.status == 'check' and node.check:
            status, dependencies = update_status(app, doctree, \
                                                 docname, [docname])
            draft_info.status = status
            draft_info.draft_dependencies = dependencies

        replacements = []
        if draft_info.status == 'yes':
            warning = create_draft_warning(draft_info.draft_dependencies)
            if node.children:
                for child in node.children:
                    warning.append(child)
            replacements.append(warning)

        node.replace_self(replacements)

    for node in doctree.traverse(refdoc_marker):
        node.replace_self([])


def setup(app):
    app.add_directive('draft', DraftNote)
    app.connect('doctree-read', process_draft_markers)
    app.connect('doctree-resolved', process_draft_nodes_resolved)
